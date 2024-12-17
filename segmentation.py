import pandas as pd
import numpy as np

class Segmentation:
    def __init__(self, df, frequency, window_size_ms, overlap_ms):
        self.df = df
        self.frequency = frequency
        self.window_size_ms = window_size_ms
        self.overlap_ms = overlap_ms

    def segment(self):
        window_size_samples = int(self.window_size_ms * self.frequency / 1000)
        overlap_samples = int(self.overlap_ms * self.frequency / 1000)
        step_size = window_size_samples - overlap_samples

        segments = []

        for start in range(0, len(self.df) - window_size_samples + 1, step_size):
            end = start + window_size_samples
            segment = self.df.iloc[start:end]
            segments.append(segment)

        return segments

    def segments_to_dataframe(self):
            """
            Convert the list of segmented DataFrames into a single DataFrame with a segment index.

            :return: A DataFrame where each row is a part of a segment, and an additional column indicates the segment index.
            """
            if not self.segments:
                raise ValueError("No segments available. Please run the 'segment' method first.")

            concatenated_segments = []
            for idx, segment in enumerate(self.segments):
                segment_copy = segment.copy()
                segment_copy['segment_index'] = idx
                concatenated_segments.append(segment_copy)

            return pd.concat(concatenated_segments, ignore_index=True)




            # this is more of a windowing
            # check jetson for inference of a small llm model 
            #using more advance machine learning mdoel 
            # using LLm voice interaction with four different cameras at different angle to guide user