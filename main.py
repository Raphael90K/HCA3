import argparse

parser = argparse.ArgumentParser(prog="Simple Neural Network Benchmakr",
                                 description="Measures the millis for a classifying neural network",
                                 epilog="Raphael Kropp")
parser.add_argument('-d', '--device', type=str, default="cpu", choices=['cpu', 'cuda', 'hailo'],
                    help="device to run the nn (default: cpu)")
parser.add_argument('-i', '--image', type=str, default="img/1.jpg",
                    help="an image to classify")

if __name__ == "__main__":
    args = parser.parse_args()
    print("Image: {}, Device: {}".format(args.image, args.device))
    if args.device == "cuda":
        import src.notebook_run as nr

        nr.use(args.image, args.device)
    elif args.device == "cpu":
        import src.notebook_run as nr

        nr.use(args.image, args.device)
    elif args.device == "hailo":
        import src.hailo_run as hr

        hr.use(args.image)
    else:
        parser.print_help()
