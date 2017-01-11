#!/usr/bin/python
# -*- coding: UTF-8 -*-

""" Time Extractor for OCR Text"""

import os.path
import traceback
import re
import json


class TimeExtractor:
    def __init__(self):
        pass
        # Load pre-trained classifier
        # self.model = joblib.load('./model/handwriting_detection_model.pkl')

    def extract_time(self, ocr_text):
        """Uses a regex pattern to extract possible times"""

        # Raise errors if any issues are identified with the input data
        if type(ocr_text) != str:
            print('ocrText field must be passed as a string.')

        def MinsToTime(_mins):
            _mins = int(_mins)
            if _mins == -1:
                return "Error"
            else:
                am_pm = "AM"
                _hr = int(_mins / 60)
            if _hr > 12:
                _hr -= 12
                am_pm = "PM"
            if _hr == 0:
                _hr += 12
                am_pm = "AM"
            elif _hr == 12:
                am_pm = "PM"
                _min = _mins % 60
                timestr = str(_hr).zfill(2) + ":" + str(_min).zfill(2) + " " + am_pm
                return timestr

        def GetTimeFromOCRString(ocrstring, verbose=False, printraw=False):
            """Only returns the first match---Should decide whether this is a good strategy or not
            AM/PM supported---24-hour format yet to be included
            returns output as minutes after midnight.
            returns -1 when there's an error
            """
            _restring = r'\b((0?[1-9]|1[012])\s?([;:,.]\s?[0-5]\s?[0-9])\s?([;:,.]\s?[0-5]\s?[0-9])?((\s)?[AaPp]\s?[mM]?)|([01]?[0-9]|2[0-3])\s?([;:,.]\s?[0-5]\s?[0-9])\s?([;:,.]\s?[0-5]\s?[0-9])?)\b'
            p = re.compile(_restring)
            try:
                multipleTimes = [(m.group(0).lower().strip(), (m.start(0), m.end(0))) for m in
                                 re.finditer(p, ocrstring) if len(m.group(0)) >= 4]
                return multipleTimes
            except:
                if verbose:
                    print("Failed to find any time")
                return -1

        def TimeStringToMins(timestring, verbose=False):
            _mins = -1
            try:
                is_am = False
                is_pm = False
                is_24 = False
                knowndelims = [';', ':', ',', '.']
                for kd in knowndelims:
                    if kd in timestring:
                        timestring = timestring.replace(kd, ":")
                        timestring = timestring.replace(" ", "")
                timestring = timestring.lower()
                if ('a' in timestring):
                    timestring = re.sub('[^a-zA-Z0-9]?a\s?m?.?', '', timestring)
                    is_am = True
                elif ('p' in timestring):
                    timestring = re.sub('[^a-zA-Z0-9]?p\s?m?.?', '', timestring)
                    is_pm = True
                else:
                    is_24 = True
                if verbose:
                    print(timestring, is_am, is_pm, is_24)
                timestring = timestring.replace(" ", ":")
                timestring = ":".join([i for i in timestring.split(":") if len(i) != 0])
                splittime = re.split(r"[^a-zA-Z0-9\s]", timestring)  # split it on anything that's a special char
                _hr, _min = int(splittime[0]), int(splittime[
                                                       1])  # we just need the hours and minutes, seconds are too granular, AM/PM is already captured
                if (is_pm) and (_hr != 12):
                    _hr += 12
                if (is_am) and (_hr == 12):
                    _hr = 0
                if verbose:
                    print(_hr * 60 + _min)
                _mins = _hr * 60 + _min
            except:
                pass
            return _mins

        def TimeStringScorer(t_pos, ocrlength, verbose=False):
            """
            Home-made scorer to see how close the extracted string is to an actual time string
            1:= highest score. There really isn't any doubt that this is a time string
            0:= lowest score. This is reserved for things that are obviously errors and not real time strings

            Factors considered:
                * Proximity to ends of the ocr string. [Turns out to be a bad ranking method]
                * Length of captured substring.
                * Penalty for looking like the price of a quantity
                * Has AM/PM explicitly in the captured group
                * TODO: penalize for stuff > 59 in amounts.
            """

            _t, _pos = t_pos  # splits the tuple into variables internally
            if (_t != "-1"):
                NUM_SCORERS = 4

                # favour strings closer to the ends of the ocr text.
                # score = 1 - normalized mean distance from any end of the string
                dist_from_ocr_beginning = (_pos[0] + _pos[1]) / float(2)
                dist_from_ocr_end = ocrlength - dist_from_ocr_beginning
                dist_score = min(dist_from_ocr_beginning, dist_from_ocr_end) / float(ocrlength)

                string_length_score = (_pos[1] - _pos[0]) / float(11)  # = len('HH:MM:SS PM'))

                # price score = 0 for looking like the price of something
                # price score = 1 for NOT looking like a price
                pricepattern = re.compile(r'\b[0-9]*\s?[.,]\s?[0-9][0-9]?\b')
                # pattern for price amount with up to 2 decimal places
                if pricepattern.match(_t):
                    price_score = 0
                else:
                    price_score = 1

                    # am_pm_score = 1 if string has either am/pm in it
                # else 1/NUM_SCORERS (doesn't affect average)
                if ('am' in _t.lower()) or ('pm' in _t.lower()):
                    am_pm_score = 1
                else:
                    am_pm_score = 1 / float(NUM_SCORERS)

                definitely_not_time_pattern = re.compile(r'\b[0-9]*[.,][6-9][0-9]?\b')
                # checks if the value in the decimal places is <=59

                scoredict = {'dist_score': dist_score, 'string_length_score': string_length_score, \
                             'price_score': price_score, 'am_pm_score': am_pm_score}
                if verbose:
                    print(_t, scoredict)
                if (definitely_not_time_pattern.match(_t)):
                    _score = 0
                    if verbose:
                        print(_t, 'definitely not a time pattern.')
                else:
                    if verbose:
                        print(_t, )
                    _score = (dist_score + string_length_score + price_score + am_pm_score) / float(NUM_SCORERS)
                return _score
            else:
                return 0

        extracted_times_positions = GetTimeFromOCRString(ocr_text)
        extracted_times = [(t[0]) for t in extracted_times_positions]
        time_mins = [TimeStringToMins(t[0]) for t in extracted_times_positions]
        timescores = [TimeStringScorer(tpos, len(ocr_text)) for tpos in extracted_times_positions]

        # Return prediction results as json
        return {'time': [{'value': time_min, 'score': score} for time_min, score in
                         sorted(zip(time_mins, timescores), key=lambda x: x[1], reverse=True)]}


if __name__ == "__main__":
    te = TimeExtractor()
    print(te.extract_time('test 1:46'))
