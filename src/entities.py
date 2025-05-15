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
    OBSERVATORY = 3


class State(Enum):
    DEFAULT = 0
    IN_REST = 1
    IN_REQUEST = 2
    IN_SECTION = 3


class Request:
    def __init__(self, telepath: int, timestamp: int):
        self.telepath = telepath
        self.timestamp = timestamp


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
        # print(f"{self.COLOR}DEBUG: [{self.RANK}] {text}")
        pass


class Observatory(Entity):
    RANK = 0
    ASTEROIDS_MIN = 1
    ASTEROIDS_MAX = 2

    def __init__(self, comm: MPI.COMM_WORLD) -> None:
        super().__init__(comm)

    def run(self) -> None:
        new_asteroids: int = 0

        while True:
            sleep(randint(3, 3 + self.SIZE))

            new_asteroids = randint(
                Observatory.ASTEROIDS_MIN, Observatory.ASTEROIDS_MAX
            )
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
        self.lock_destroyed_asteroids = Lock()
        self.has_every_ack = Condition()
        self.might_be_first_in_queue = Condition()
        self.received_asteroids = Condition()

        # Local variables
        self.request_queue: list[Request] = []
        self.lamport_clock = 0
        self.destroyed_asteroids = 0
        self.asteroids_count = 0
        self.ack_count = 1

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
                    self.debug("Sending requests")
                    self._send_requests()

                    self.print("Waiting to be assigned an asteroid")
                    self.has_every_ack.wait()  # Wait to be notified
                    self.debug(f"Got all ACKs ({self.ack_count})")

                # Stop from advancing unless first in queue
                while True:
                    with self.might_be_first_in_queue:
                        # Check if first in queue
                        with self.lock_request_queue:
                            if self.request_queue[0].telepath == self.RANK:
                                self.debug("I'm first in queue")
                                break
                        self.debug("Waiting to be first in queue")
                        self.might_be_first_in_queue.wait()  # Wait to be notified
                state = State.IN_SECTION

            elif state == State.IN_SECTION:
                # Stop from advancing unless an asteroid is available
                while True:
                    with self.received_asteroids:
                        # Check if an asteroid is available
                        with self.lock_destroyed_asteroids:
                            if self.asteroids_count > self.destroyed_asteroids:
                                break
                        self.debug(
                            f"Waiting for an asteroid to be available [{self.destroyed_asteroids}/{self.asteroids_count}]"
                        )
                        self.received_asteroids.wait()

                with self.lock_destroyed_asteroids:
                    self.destroyed_asteroids += 1
                    self.print(f"Destroying the asteroid [{self.destroyed_asteroids}]")

                sleep(self.WORK_TIME)  # Simulate work

                # self.print("The asteroid was destroyed")
                # sleep(0.1) # Prevent the print from happening after release

                self._remove_from_queue(self.RANK)
                self.debug("Sending release")
                self._send_release()
                self.ack_count = 1

                state = State.DEFAULT

            else:
                self.print(f"Entered an incorrect state [{state}]")
                state = State.DEFAULT

    def _handle_communication(self) -> None:
        status = MPI.Status()

        while True:
            data: Message | Asteroids_Info = self.COMM.recv(
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

                if self.ack_count == self.SIZE - 1:
                    with self.has_every_ack:
                        self.has_every_ack.notify()
                        self.debug("Notified has_every_ack condition")

            elif status.tag == Tag.RELEASE.value:
                self.debug(f"Received RELEASE from {status.source}")
                with self.lock_destroyed_asteroids:
                    self.destroyed_asteroids += 1

                self._remove_from_queue(status.source)
                self.debug(f"Removed {status.source} from queue")

                with self.might_be_first_in_queue:
                    self.might_be_first_in_queue.notify()
                    self.debug("Notified might_be_first_in_queue condition")

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

    def _send_release(self) -> None:
        with self.lock_lamport_clock:
            self.lamport_clock += 1

            for rank in range(self.SIZE):
                if rank == self.RANK or rank == Observatory.RANK:
                    continue
                else:
                    self.COMM.send(
                        Message(self.lamport_clock),
                        dest=rank,
                        tag=Tag.RELEASE.value,
                    )

    def _send_requests(self) -> None:
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

    def _remove_from_queue(self, telepath: int) -> None:
        with self.lock_request_queue:
            self.request_queue = [
                req for req in self.request_queue if req.telepath != telepath
            ]
