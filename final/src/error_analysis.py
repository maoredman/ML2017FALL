import numpy as np
import pickle
import sys
def main():
    question_id = [-1]
    question_text = [sys.argv[1]]
    segment_start_position = [0]
    segment_text = [sys.argv[2]]
    pickle.dump(question_id, open("./data/error_analyze/question_id.pickle", "wb"))
    pickle.dump(question_text, open("./data/error_analyze/question_text.pickle", "wb"))
    pickle.dump(segment_start_position, open("./data/error_analyze/segment_start_position.pickle", "wb"))
    pickle.dump(segment_text, open("./data/error_analyze/segment_text.pickle", "wb"))
if __name__ == '__main__':
    main()