import argparse

from generate_polygons import polygonize

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Postprocessing")
    arg = parser.add_argument
    arg('--masks-path', type=str, default='../test_spacenet/ensemble', help='Path to predicted masks')
    arg('--output-path', type=str, help='Path for output file', default="submission.csv")

    args = parser.parse_args()


    polygonize(args.masks_path, args.output_path)
