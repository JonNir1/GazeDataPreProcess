from DataParser.TobiiGazeDataParser import TobiiGazeDataParser


def calculate_sampling_rate_for_tobii(tobii_path) -> float:
    # Returns the actual sampling rate of the Tobii data.
    tobii_parser = TobiiGazeDataParser(tobii_path)
    return tobii_parser.sampling_rate

