def print_dashes():
    print("-" * 80)


def print_equal():
    print("=" * 80)


def print_analysis_type(pipeline_type, dim):
    print("\n")
    print_equal()
    print_text_centered(f"{pipeline_type.value}")
    print_text_centered(f"{dim}D")
    print_equal()
    print("\n")


def print_text_centered(text: str):
    print(f"{' ' * ((80 - len(text)) // 2)}{text}")


def print_text_inside(text: str, car: str):
    print("\n")
    print(
        f"{car * ((80 - len(text)) // 2 - 1)} {text} {car * ((80 - len(text)) // 2 - 1)}"
    )


def print_routine(name):
    print("_" * (len(name) + 2))
    print(f" {name} \n")
