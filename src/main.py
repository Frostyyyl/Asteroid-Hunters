from mpi4py import MPI
from entities import Observatory, Telepath


def main():
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 1:
        observatory = Observatory(comm)
        observatory.run()
    else:
        telepath = Telepath(comm)
        telepath.run()


if __name__ == '__main__':
    main()
