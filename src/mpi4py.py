class MPI:
    class COMM_WORLD:
        def Get_rank(self) -> int:
            return 0

        def Get_size(self) -> int:
            return 1

    def Init_thread(self) -> int:
        return 0

    def Finalize() -> None:
        return

    COMM_WORLD = COMM_WORLD()
    THREAD_SERIALIZED = int(0)
