from time import sleep
from colors import Colors
from mpi4py import MPI
from threading import Thread, Lock, Condition
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
    def __init__(self, timestamp, sender):
        self.timestamp = timestamp
        self.sender = sender


class Observatorys_Notify:
    def __init__(self, detected_asteroids):
        self.detected_asteroids = detected_asteroids


class Request:
    def __init__(self, applicant, timestamp):
        self.applicant = applicant
        self.timestamp = timestamp


class Entity:
    def __init__(self, comm: MPI.COMM_WORLD) -> None:
        self.COMM = comm
        self.SIZE = comm.Get_size()
        self.RANK = comm.Get_rank()
        self.COLOR = Colors.get_color(self.RANK)

    def print(self, text) -> None:
        print(f'{self.COLOR}[{self.RANK}] {text}')


class Observatory(Entity):
    def __init__(self, comm):
        super().__init__(comm)

    def run(self) -> None:
        while True:
            sleep(2)
            self.print('Just slept')


class Telepath(Entity):
    # NOTE: Any and all variables shared between communication thread
    # (messenger) and main loop (run) must be protected by using
    # a Lock if using a read and assign operators (+=, -=, /=, *=).
    # Not necessary only if just one side is ever writing

    def __init__(self, comm) -> None:
        super().__init__(comm)

        # Local time variables
        self.WORK_TIME = 3
        self.SLEEP_TIME = randint(1, comm.Get_size())

        # Variables used in threading
        self.messenger = Thread(target=self._handle_communication, daemon=True)
        self.lock_lamport_clock = Lock()
        self.has_all_ack = Condition()

        # Local variables
        self.request_queue: list[Request] = []
        self.lamport_clock = 0
        self.asteroids_count = 0
        self.served_asteroids = 0
        self.ack_count = 0
        self.state = State.DEFAULT

    def _handle_communication(self) -> None:
        status = MPI.Status()

        while True:
            data: Message | Observatorys_Notify = self.COMM.recv(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
            )

            if status.tag != Tag.OBSERVATORY:
                with self.lock_lamport_clock:
                    self.lamport_clock = max(self.lamport_clock, data.timestamp) + 1

            match status.tag:
                case Tag.REQ:
                    self.request_queue.append(Request(data.sender, data.timestamp))
                    self.request_queue.sort(key=lambda x: (x.timestamp, x.applicant))
                    self.COMM.send(
                        Message(self.lamport_clock, self.RANK),
                        dest=data.sender,
                        tag=Tag.ACK,
                    )

                case Tag.ACK:
                    self.ack_count += 1
                    if self.ack_count == self.SIZE - 1:
                        self.has_all_ack.notify()

                case Tag.RELEASE:
                    self.served_asteroids += 1
                    self.request_queue = [
                        req
                        for req in self.request_queue
                        if req.applicant != data.sender
                    ]
                    if self.is_requesting and self._can_enter_critical_section():
                        self._enter_critical_section()

                case Tag.OBSERVATORY:
                    self.asteroids_count += data.detected_asteroids

                case _:
                    self.print(f'Received unknown message tag [{status.tag}]')

    def _send_release(self) -> None:
        with self.lock_lamport_clock:
            self.lamport_clock += 1

            for rank in range(self.SIZE):
                if rank != self.RANK:
                    self.COMM.send(
                        Message(self.lamport_clock, self.RANK),
                        dest=rank,
                        tag=Tag.RELEASE,
                    )

    def _send_request(self) -> None:
        with self.lock_lamport_clock:
            self.lamport_clock += 1
            self.request_queue.append(Request(self.RANK, self.lamport_clock))

            for rank in range(self.SIZE):
                if rank != self.RANK:
                    self.COMM.send(
                        Message(self.lamport_clock, self.RANK), dest=rank, tag=Tag.REQ
                    )

    def run(self) -> None:
        self.messenger.start()
        just_rested = False

        while True:
            match self.state:
                case State.DEFAULT:
                    if not just_rested and randint(1, 100) <= 50:
                        self.state = State.IN_REST
                    else:
                        just_rested = False
                        self._send_request()
                        self.state = State.IN_REQUEST

                case State.IN_REST:
                    self.print('I\'m taking a break')
                    just_rested = True
                    sleep(self.SLEEP_TIME)
                    self.state = State.DEFAULT

                case State.IN_REQUEST:
                    self.print('Waiting to be assigned an asteroid')

                    with self.has_all_ack:
                        self.has_all_ack.wait()  # Wait to be notified
                        self.state = State.IN_SECTION

                case State.IN_SECTION:
                    self.print("Destroying an asteroid")
                    self.served_asteroids += 1
                    sleep(self.WORK_TIME)

                    self._send_release()
                    self.ack_count = 0

                    # Remove our own request from the queue
                    self.request_queue = [
                        req for req in self.request_queue if req.applicant != self.RANK
                    ]
                    self.state = State.DEFAULT

                case _:
                    self.print(f'Entered an incorrect state [{self.state}]')
                    self.state = State.DEFAULT
