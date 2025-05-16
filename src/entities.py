from threading import Thread, Lock, Condition
from random import randint
from colors import Colors
from time import sleep
from mpi4py import MPI
from enum import Enum


class Tag(Enum):
    REQ = 0
    ACK = 1
    RELEASE = 2
    PAIR = 3
    OBSERVATORY = 4


class State(Enum):
    DEFAULT = 0
    IN_REST = 1
    IN_REQUEST = 2
    IN_SECTION = 3


class PairState(Enum):
    DEFAULT = 0
    IS_FIRST = 1
    IS_SECOND = 2
    FINISHED = 3


class Request:
    def __init__(self, telepath: int, timestamp: int):
        self.telepath = telepath
        self.timestamp = timestamp


Release = Request


class Message:
    def __init__(self, timestamp: int):
        self.timestamp = timestamp


class Asteroids_Info:
    def __init__(self, new_asteroids: int):
        self.new_asteroids = new_asteroids


class Entity:
    def __init__(self, comm: MPI.COMM_WORLD) -> None:
        self.COMM = comm
        self.SIZE = comm.Get_size()
        self.RANK = comm.Get_rank()
        self.COLOR = Colors.get_color(self.RANK)

    def print(self, text: str) -> None:
        print(f"{self.COLOR}[{self.RANK}] {text}")
        pass

    def debug(self, text: str) -> None:
        print(f"{self.COLOR}DEBUG: [{self.RANK}] {text}")
        pass


class Observatory(Entity):
    # NOTE: There should be only one observatory ever with RANK 0
    RANK = 0  # Used inside telepath processes
    ASTEROIDS_MIN = 1
    ASTEROIDS_MAX = 2

    def __init__(self, comm: MPI.COMM_WORLD) -> None:
        super().__init__(comm)

    def run(self) -> None:
        new_asteroids: int = 0

        while True:
            sleep(randint(3, 3 + self.SIZE))

            new_asteroids = randint(self.ASTEROIDS_MIN, self.ASTEROIDS_MAX)
            self.print(f"Found {new_asteroids} new asteroids, sending notice")

            for rank in range(self.SIZE):
                if rank != self.RANK:
                    self.COMM.send(
                        Asteroids_Info(new_asteroids),
                        dest=rank,
                        tag=Tag.OBSERVATORY.value,
                    )


class Telepath(Entity):
    # NOTE: Any and all variables shared between communication thread
    # (messenger) and main loop (run) must be protected using
    # a Lock if using a read and assign operators (+=, -=, /=, *=).

    def __init__(self, comm: MPI.COMM_WORLD) -> None:
        super().__init__(comm)

        # Local time variables
        self.WORK_TIME = 3
        self.SLEEP_TIME = randint(2, 2 + self.SIZE)

        # NOTE: asteroids_count does not need a lock because only
        # messenger is ever modifying it
        # NOTE: ack_count does not need a lock because of
        # when each thread accesses it

        # Variables used in threading
        self.messenger = Thread(target=self._handle_communication, daemon=True)
        self.lock_request_queue = Lock()
        self.lock_lamport_clock = Lock()
        self.has_every_ack = Condition()
        self.might_be_first_in_queue = Condition()
        self.received_asteroids = Condition()
        self.got_pair_info = Condition()

        # Local variables
        self.request_queue: list[Request] = []
        self.lamport_clock = 0
        self.ack_count = 1
        self.destroyed_asteroids = 0  # NOTE: Use with lock_request_queue
        self.assigned_asteroid = -1
        self.asteroids_count = 0
        self.pair_state = PairState.DEFAULT
        self.pair_number = -1

    def run(self) -> None:
        self.messenger.start()
        state = State.DEFAULT
        just_rested = False

        while True:
            if state == State.DEFAULT:
                # Choose between sleeping and working
                if not just_rested and randint(1, 100) <= 50:
                    state = State.IN_REST
                else:
                    just_rested = False
                    state = State.IN_REQUEST

            elif state == State.IN_REST:
                self.print("Taking a break")
                just_rested = True
                sleep(self.SLEEP_TIME)
                state = State.DEFAULT

            elif state == State.IN_REQUEST:
                # In order to prevent a deadlock the Condition must
                # be aquired before sending requests
                with self.has_every_ack:
                    self._send_requests_and_add_to_queue()

                    self.print("Waiting to be assigned an asteroid")
                    self.has_every_ack.wait()  # Wait to be notified
                    self.debug(f"Got all ACKs ({self.ack_count})")

                # Notify or be notified by your pair
                with self.lock_request_queue:
                    for i, val in enumerate(self.request_queue):
                        if val.telepath == self.RANK:
                            self.debug
                            if (i + 1) % 2 == 0:
                                self.pair_state = PairState.IS_SECOND
                                self.pair_number = self.request_queue[i - 1].telepath
                                self._notify_pair()
                                self.assigned_asteroid = (
                                    self.destroyed_asteroids + 1 + (i // 2)
                                )
                                break
                            else:
                                # The lock_request_queue must be released
                                # if no matching pair is currently available,
                                # to allow new information to be received
                                # and added to the queue.
                                break

                # The first's thread info is assigned inside
                # the messenger thread
                if self.pair_state != PairState.IS_SECOND:
                    while True:
                        with self.got_pair_info:
                            if self.pair_state == PairState.IS_FIRST:
                                # Aquire the assigned asteroid number
                                with self.lock_request_queue:
                                    for i, val in enumerate(self.request_queue):
                                        if val.telepath == self.RANK:
                                            self.assigned_asteroid = (
                                                self.destroyed_asteroids + 1 + (i // 2)
                                            )
                                break  # Move forward
                            self.debug("Waiting to be assigned a pair")
                            self.got_pair_info.wait()

                state = State.IN_SECTION

            elif state == State.IN_SECTION:
                # Stop from advancing unless my asteroid is available
                while True:
                    # Lock in advance to not lose the nofity
                    with self.received_asteroids:
                        # Check if the asteroid is available
                        if self.assigned_asteroid <= self.asteroids_count:
                            self.debug("An asteroid is available")
                            break  # Move forward
                        self.debug(
                            f"Waiting for an asteroid to be available [{self.assigned_asteroid}/{self.asteroids_count}]"
                        )
                        self.received_asteroids.wait()

                self.print(f"Destroying the asteroid [{self.assigned_asteroid}]")

                sleep(self.WORK_TIME)  # Simulate work

                if self.pair_state == PairState.IS_FIRST:
                    self._notify_pair()  # Notify its pair that it has finished
                    while True:
                        # Wait for reply
                        with self.got_pair_info:
                            if self.pair_state == PairState.FINISHED:
                                break
                            self.got_pair_info.wait()
                else:
                    while True:
                        # Wait for its pair to finish
                        with self.got_pair_info:
                            if self.pair_state == PairState.FINISHED:
                                self.debug("Sending release")
                                self._send_release()
                                self._remove_from_queue(self.RANK, self.pair_number)
                                break
                            self.got_pair_info.wait()

                self.debug("Job finished")
                self.ack_count = 1
                state = State.DEFAULT

            else:
                self.print(f"Entered an incorrect state [{state}]")
                state = State.DEFAULT

    def _handle_communication(self) -> None:
        status = MPI.Status()

        while True:
            data: Message | Asteroids_Info | Release = self.COMM.recv(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
            )

            if status.tag != Tag.OBSERVATORY.value:
                # Update the lamport's clock
                with self.lock_lamport_clock:
                    self.lamport_clock = max(self.lamport_clock, data.timestamp) + 1
                    self.debug(f"Updated Lamport's clock: {self.lamport_clock}")

            if status.tag == Tag.REQ.value:
                self.debug(f"Received REQ from {status.source}")
                self._add_to_queue(status.source, data.timestamp)

                # Send a confirmation
                with self.lock_lamport_clock:
                    self.COMM.send(
                        Message(self.lamport_clock),
                        dest=status.source,
                        tag=Tag.ACK.value,
                    )
                    self.debug(f"Sent ACK to {status.source}")

            elif status.tag == Tag.ACK.value:
                self.debug(f"Received ACK from {status.source}")
                self.ack_count += 1

                # Notify about receiving all ACKs
                if self.ack_count == self.SIZE - 1:
                    with self.has_every_ack:
                        self.has_every_ack.notify()
                        self.debug("Notified has_every_ack condition")

            elif status.tag == Tag.RELEASE.value:
                self.debug(f"Received RELEASE from {status.source}")
                self._remove_from_queue(status.source, data.telepath)

                # Check if the received message includes me
                # if so the job is finished (Look: State.IN_SECTION)
                if data.telepath == self.RANK:
                    with self.got_pair_info:
                        self.pair_state = PairState.FINISHED
                        self.got_pair_info.notify()
                        self.debug("Notified got_pair_info condition")
                else:
                    with self.might_be_first_in_queue:
                        self.might_be_first_in_queue.notify()
                        self.debug("Notified might_be_first_in_queue condition")

            elif status.tag == Tag.PAIR.value:
                self.debug(f"Received PAIR from {status.source}")
                if self.pair_state == PairState.IS_SECOND:
                    self.pair_state = PairState.FINISHED
                else:
                    self.pair_number = status.source
                    self.pair_state = PairState.IS_FIRST
                with self.got_pair_info:
                    self.got_pair_info.notify()

            elif status.tag == Tag.OBSERVATORY.value:
                self.asteroids_count += data.new_asteroids
                self.debug(
                    f"Added {data.new_asteroids} asteroids (total: {self.asteroids_count})"
                )
                with self.received_asteroids:
                    self.received_asteroids.notify()
                    self.debug("Notified received_asteroids condition")

            else:
                self.print(
                    f"Received unknown message tag [{status.tag}] from {status.source}"
                )

    def _notify_pair(self) -> None:
        self.debug(f"Sending notice to my pair [{self.pair_number}]")
        with self.lock_lamport_clock:
            self.lamport_clock += 1

            for rank in range(self.SIZE):
                if rank == self.pair_number:
                    self.COMM.send(
                        Message(self.lamport_clock),
                        dest=rank,
                        tag=Tag.PAIR.value,
                    )

    def _send_release(self) -> None:
        with self.lock_lamport_clock:
            self.lamport_clock += 1

            for rank in range(self.SIZE):
                if rank == self.RANK or rank == Observatory.RANK:
                    continue
                else:
                    self.COMM.send(
                        Release(self.pair_number, self.lamport_clock),
                        dest=rank,
                        tag=Tag.RELEASE.value,
                    )

    def _send_requests_and_add_to_queue(self) -> None:
        self.debug("Sending requests")
        with self.lock_lamport_clock:
            self.lamport_clock += 1

            self._add_to_queue(self.RANK, self.lamport_clock)

            for rank in range(self.SIZE):
                if rank == self.RANK or rank == Observatory.RANK:
                    continue
                else:
                    self.COMM.send(
                        Message(self.lamport_clock),
                        dest=rank,
                        tag=Tag.REQ.value,
                    )

    def _add_to_queue(self, telepath: int, timestamp: int) -> None:
        with self.lock_request_queue:
            self.request_queue.append(Request(telepath, timestamp))
            self.request_queue.sort(key=lambda x: (x.telepath))
            self.request_queue.sort(key=lambda x: (x.timestamp))
            self.debug(
                f"Added {telepath} to queue. Request queue: {[(r.telepath, r.timestamp) for r in self.request_queue]}"
            )

    def _remove_from_queue(self, telepath: int, telepath_pair: int) -> None:
        """
        Removes a request from the queue and increases the number of
        destroyed asteroids
        """
        with self.lock_request_queue:
            self.destroyed_asteroids += 1
            self.request_queue = [
                req
                for req in self.request_queue
                if (req.telepath != telepath and req.telepath != telepath_pair)
            ]
        self.debug(f"Removed {telepath} and {telepath_pair} from queue")
