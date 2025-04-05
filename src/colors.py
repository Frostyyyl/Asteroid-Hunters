from colorama import Fore


class Colors:
    _colors = (
        Fore.CYAN,
        Fore.LIGHTGREEN_EX,
        Fore.LIGHTRED_EX,
        Fore.LIGHTWHITE_EX,
        Fore.LIGHTBLUE_EX,
        Fore.BLUE,
    )

    @staticmethod
    def get_color(id: int) -> str:
        return Colors._colors[id % len(Colors._colors)]
