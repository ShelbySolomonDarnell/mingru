
@staticmethod
def splitFileThenWrite(path: str, split_percentage: int):
    """Splits a file based on the split_percentage into two files."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    n = len(data)
    the_result   = data[: int(n * (split_percentage/100))]
    val_leftover = data[int(n * (split_percentage/100)) :]

    name_a = path + "." + str(split_percentage) + "percent"
    name_b = path + "." + str(100-split_percentage) + "percent"

    try:
        with open(name_a, "x") as f:
            f.write(the_result)
        with open(name_b, "x") as f:
            f.write(val_leftover)
    except FileExistsError:
        print("I am not authorized to overwrite files...!")


if __name__ ==  "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("textfile", help="Path to file that needs splitting")
    parser.add_argument("--split", type=int, default=10)
    args = vars(parser.parse_args())
    print("Textfile {}\nSplit percentage {}".format(args["textfile"], args["split"]))
    splitFileThenWrite(args["textfile"], args["split"])
