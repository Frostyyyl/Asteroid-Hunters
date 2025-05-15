from time import sleep
from colors import Colors
from mpi4py import MPI
from threading import Thread, Lock, RLock, Condition
from random import randint
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


class Message:
    def __init__(self, telepath: int, timestamp: int):
        self.timestamp = timestamp
        self.telepath = telepath


Request = Message


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


class Observatory(Entity):
    RANK = 0
    ASTEROIDS_MIN = 1
    ASTEROIDS_MAX = 3

    def __init__(self, comm: MPI.COMM_WORLD) -> None:
        super().__init__(comm)

    def run(self) -> None:
        new_asteroids: int = 0

        while True:
            sleep(randint(4, 4 + self.SIZE))

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
    # (messenger) and main loop (run) must be protected by using
    # a Lock if using a read and assign operators (+=, -=, /=, *=).

    def __init__(self, comm: MPI.COMM_WORLD) -> None:
        super().__init__(comm)

        # Local time variables
        self.WORK_TIME = 3
        self.SLEEP_TIME = randint(1, self.SIZE)

        # Variables used in threading
        # NOTE: asteroids_count does not need a lock because only
        # messenger is ever modifying it
        # NOTE: ack_count does not need a lock because of
        # when each thread accesses it
        self.messenger = Thread(target=self._handle_communication, daemon=True)
        self.lock_request_queue = Lock()
        self.lock_lamport_clock = Lock()
        self.lock_destroyed_asteroids = Lock()
        self.has_every_ack = Condition()
        self.is_first_in_queue = Condition()
        self.has_an_asteroid = Condition()

        # Local variables
        self.request_queue: list[Request] = []
        self.lamport_clock = 0
        self.asteroids_count = 0
        self.destroyed_asteroids = 0
        self.ack_count = 1

    def debug(self, text: str) -> None:
        print(f"{self.COLOR}DEBUG: [{self.RANK}] {text}")
        pass

    def _handle_communication(self) -> None:
        status = MPI.Status()
        while True:
            data: Message | Asteroids_Info = self.COMM.recv(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
            )

            if status.tag != Tag.OBSERVATORY.value:
                with self.lock_lamport_clock:
                    self.lamport_clock = max(self.lamport_clock, data.timestamp) + 1
                    self.debug(f"Updated Lamport's clock: {self.lamport_clock}")

            if status.tag == Tag.REQ.value:
                self.debug(f"Received REQ from {data.telepath}")
                with self.lock_request_queue:
                    self.request_queue.append(Request(data.telepath, data.timestamp))
                    self.request_queue.sort(key=lambda x: (x.telepath))
                    self.request_queue.sort(key=lambda x: (x.timestamp))
                    self.debug(
                        f"Request queue: {[(r.telepath, r.timestamp) for r in self.request_queue]}"
                    )

                with self.lock_lamport_clock:
                    self.COMM.send(
                        Message(self.RANK, self.lamport_clock),
                        dest=data.telepath,
                        tag=Tag.ACK.value,
                    )
                    self.debug(f"Sent ACK to {status.source}")

            elif status.tag == Tag.ACK.value:
                self.debug(f"Received ACK from {status.source}")
                self.ack_count += 1
                with self.has_every_ack:
                    if self.ack_count == self.SIZE - 1:
                        self.has_every_ack.notify()
                        self.debug("Notified has_every_ack condition")

            elif status.tag == Tag.RELEASE.value:
                self.debug(f"Received RELEASE from {data.telepath}")
                with self.lock_destroyed_asteroids:
                    self.destroyed_asteroids += 1

                self._remove_from_request_queue(data.telepath)
                self.debug(f"Removed {status.source} from queue")
                with self.is_first_in_queue:
                    self.is_first_in_queue.notify()
                    self.debug("Notified is_first_in_queue condition")

            elif status.tag == Tag.OBSERVATORY.value:
                self.asteroids_count += data.new_asteroids
                with self.has_an_asteroid:
                    self.has_an_asteroid.notify()
                self.debug(
                    f"Added {data.new_asteroids} asteroids (total: {self.asteroids_count})"
                )

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
                        Message(self.RANK, self.lamport_clock),
                        dest=rank,
                        tag=Tag.RELEASE.value,
                    )

    def _send_requests(self) -> None:
        with self.lock_lamport_clock:
            self.lamport_clock += 1
            with self.lock_request_queue:
                self.request_queue.append(Request(self.RANK, self.lamport_clock))

            for rank in range(self.SIZE):
                if rank == self.RANK or rank == Observatory.RANK:
                    continue
                else:
                    self.COMM.send(
                        Message(self.RANK, self.lamport_clock),
                        dest=rank,
                        tag=Tag.REQ.value,
                    )

    def _remove_from_request_queue(self, telepath: int) -> None:
        with self.lock_request_queue:
            self.request_queue = [
                req for req in self.request_queue if req.telepath != telepath
            ]

    def run(self) -> None:
        self.messenger.start()
        state = State.DEFAULT
        just_rested = False

        while True:
            if state == State.DEFAULT:
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
                    self._send_requests()

                    self.print("Waiting to be assigned an asteroid")
                    self.has_every_ack.wait()  # Wait to be notified
                    self.debug(f"Got all ACKs ({self.ack_count})")

                    while True:
                        with self.is_first_in_queue:
                            with self.lock_request_queue:
                                if self.request_queue[0].telepath == self.RANK:
                                    self.debug("I'm first in queue")
                                    break
                            self.debug("Waiting to be first in queue")
                            self.is_first_in_queue.wait()

                    state = State.IN_SECTION

            elif state == State.IN_SECTION:
                while True:
                    with self.has_an_asteroid:
                        with self.lock_destroyed_asteroids:
                            if self.asteroids_count > self.destroyed_asteroids:
                                break
                        self.debug(
                            f"Waiting for an asteroid notify [{self.asteroids_count}/{self.destroyed_asteroids}]"
                        )
                        self.has_an_asteroid.wait()

                self.print("Destroying an asteroid")
                with self.lock_destroyed_asteroids:
                    self.destroyed_asteroids += 1
                    self.debug(f"Destroyed asteroids: {self.destroyed_asteroids}")
                sleep(self.WORK_TIME)

                self.print("The asteroid was destroyed")

                self._remove_from_request_queue(self.RANK)
                self._send_release()
                self.ack_count = 1

                state = State.DEFAULT

            else:
                self.print(f"Entered an incorrect state [{state}]")
                state = State.DEFAULT
