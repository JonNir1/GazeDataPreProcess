from DataParser.TobiiGazeDataParser import TobiiGazeDataParser


path = r"C:\Users\jonathanni\Desktop\b.txt"
tobii_parser = TobiiGazeDataParser(path)
df = tobii_parser.parse_gaze_data()
