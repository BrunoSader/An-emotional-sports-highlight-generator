#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from matplotlib import pyplot as plt

import numpy as np
import pytesseract
import cv2
import tkinter as tk
import logging
import time
import re
import threading


# Dimensioning values
# We are defining global variables based on match data in order to isolate the scoreboard
left_x = 170
upper_y = 45
right_x = 500
lower_y = 80

time_divide = 240
time_width = 65
time_position = 'left'
# If time-position = right : scoreboard is on the left and time on the right
# Else if time position = left : scoreboard is on the right and time on the left

# To deal with time.sleep() and effectively end the threads
#time_value = 0

class ImageHandler(object):

    def __init__(self, export_path, filename_in):
        self.scoreboard_image = None
        self.time_image = None
        self.time_text = None
        self.teams_goals_image = None
        self.teams_goals_text = None

        self.video_source_path = filename_in
        self.export_image_path = export_path + '/football.jpg'
        self.export_path = export_path

        logging.basicConfig(level=logging.WARNING)

    def extract_image_from_video(self):
        """
        Extracts image from video and saves on disk with specified period.

        :param path_to_video: Path to video and video name with file format
        :param export_image_path: Export image path and image name with file format
        :return: -
        """

        vidcap = cv2.VideoCapture(self.video_source_path)
        count = 0
        #success = True
        image_lst = []

        while(True):
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
            success, image = vidcap.read()
            image_lst.append(image)

            # Stop when last frame is identified
            if count > 1:
                if np.array_equal(image, image_lst[1]):
                    break
                image_lst.pop(0)  # Clean the list
                # save frame as PNG file

            if(ocr.index < ocr.video_length):
                try:
                    cv2.imwrite(self.export_image_path, image)
                    print('{}.sec reading a new frame: {} '.format(count, success))
                    count += 1
                    ocr.eImageExported.set()
                    time.sleep(1)
                except Exception as e:
                    pass

    def localize_scoreboard_image(self):
        """
        Finds the scoreboard table in the upper corner, sets scoreboard_image
        and exports the picture as 'scoreboard_table.jpg'

        :return: True when scoreboard is found
                 False when scoreboard is not found
        """

        # Read a snapshot image from the video and convert to gray
        snapshot_image = cv2.imread(self.export_image_path)
        grayscale_image = cv2.cvtColor(snapshot_image, cv2.COLOR_BGR2GRAY)

        self.scoreboard_image = grayscale_image[upper_y:lower_y,
                                                left_x:right_x]
        cv2.imwrite(self.export_path + '/scoreboard_table.jpg',
                    self.scoreboard_image)

    def split_scoreboard_image(self):
        """
        Splits the scoeboard image into two parts, sets 'time_image' and 'teams_goals_image'
        and exports as 'time_table.jpg' and 'teams_goals_table.jpg'
        Left image represents the time.
        Right image represents the teams and goals.

        :return: -
        """

        '''
        self.time_image = self.scoreboard_image[:, 0:175]
        cv2.imwrite('ocr/img/time_table.jpg', self.time_image)

        self.teams_goals_image = self.scoreboard_image[:, 175:]
        cv2.imwrite('ocr/img/teams_goals_table.jpg', self.teams_goals_image)
        '''

        relative_time_divide = time_divide-left_x
        time_end = relative_time_divide + time_width

        if(time_position == 'right'):
            self.time_image = self.scoreboard_image[:,
                                                    relative_time_divide:time_end]
            cv2.imwrite(self.export_path + '/time_table.jpg', self.time_image)

            self.teams_goals_image = self.scoreboard_image[:,
                                                           0:relative_time_divide]
            cv2.imwrite(self.export_path + '/teams_goals_table.jpg',
                        self.teams_goals_image)

        else:
            self.time_image = self.scoreboard_image[:, 0:relative_time_divide]
            cv2.imwrite(self.export_path + '/time_table.jpg', self.time_image)

            self.teams_goals_image = self.scoreboard_image[:,
                                                           relative_time_divide:]
            cv2.imwrite(self.export_path + '/teams_goals_table.jpg',
                        self.teams_goals_image)

    def enlarge_scoreboard_images(self, enlarge_ratio):
        """
        Enlarges 'time_table.jpg' and 'teams_goals_table.jpg'

        :param enlarge_ratio: Defines the enlarging size (e.g 2-3x)
        :return: -
        """
        self.time_image = cv2.resize(
            self.time_image, (0, 0), fx=enlarge_ratio, fy=enlarge_ratio)
        self.teams_goals_image = cv2.resize(
            self.teams_goals_image, (0, 0), fx=enlarge_ratio, fy=enlarge_ratio)

    def _get_time_from_image(self):
        """
        Preprocesses time_image transformations for OCR.
        Exports 'time_ocr_ready.jpg' after the manipulations.
        Reads match time from 'time_ocr_ready.jpg' using Tesseract.
        Applies result to time_text.

        :return: True: string is found
                 False: string is not found
        """

        # Count nonzero to determine contrast type
        ret, threshed_img = cv2.threshold(
            self.time_image, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

        self.time_image = cv2.GaussianBlur(self.time_image, (3, 3), 0)

        kernel = np.ones((3, 3), np.uint8)

        self.time_image = cv2.erode(self.time_image, kernel, iterations=1)
        self.time_image = cv2.dilate(self.time_image, kernel, iterations=1)

        cv2.imwrite(self.export_path + '/time_ocr_ready.jpg', self.time_image)

        self.time_text = pytesseract.image_to_string(
            Image.open(self.export_path + '/time_ocr_ready.jpg'), config="--psm 6")
        logging.info('Time OCR text: {}'.format(self.time_text))

        if self.time_text is not None:
            return True
        return False

    def _get_teams_goals_from_image(self):
        """
        Preprocesses teams_goals_image with transformations for OCR.
        Exports 'teams_goals_ocr_ready.jpg' after the manipulations.
        Reads teams and goals information from 'teams_goals_ocr_ready.jpg' using Tesseract.
        Applies result to teams_goals_text.

        :return: True: string is found
                 False: string is not found

        """
        # Applying Thresholding for Teams goals OCR preprocess
        ret, self.teams_goals_image = cv2.threshold(
            self.teams_goals_image, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

        self.teams_goals_image = cv2.GaussianBlur(
            self.teams_goals_image, (3, 3), 0)

        kernel = np.ones((3, 3), np.uint8)

        #self.teams_goals_image = cv2.erode(self.teams_goals_image, kernel, iterations=1)
        self.teams_goals_image = cv2.dilate(
            self.teams_goals_image, kernel, iterations=1)

        cv2.imwrite(self.export_path + '/teams_goals_ocr_ready.jpg',
                    self.teams_goals_image)

        self.teams_goals_text = pytesseract.image_to_string(
            Image.open(self.export_path + '/teams_goals_ocr_ready.jpg'))
        logging.info('Teams and goals OCR text: {}'.format(
            self.teams_goals_text))

        if self.teams_goals_text is not None:
            return True
        return False

    def get_scoreboard_texts(self):
        """
        Returns an array of strings including OCR read time, teams and goals texts.
        :return: numpy array 'scoreboard_texts'
                 scoreboard_texts[0] : time text value
                 scoreboard_texts[1] : teams and goals text value

        """
        # Read text values using Tesseract OCR
        time_text_exists = self._get_time_from_image()
        teams_goals_text_exists = self._get_teams_goals_from_image()

        scoreboard_texts = []
        # Use values on successful read
        if time_text_exists and teams_goals_text_exists:
            scoreboard_texts.append(self.time_text)
            scoreboard_texts.append(self.teams_goals_text)
            scoreboard_texts = np.array(scoreboard_texts)

        return scoreboard_texts

    def play_match_video(self):

        cap = cv2.VideoCapture(self.video_source_path)
        count = 0

        if(ocr.index < ocr.video_length):
            while (cap.isOpened()):
                cap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))
                ret, frame = cap.read()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                cv2.imshow('frame', gray)
                time.sleep(1)
                count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


class Match(object):

    def __init__(self, export_path, filename_out):
        self.scoreboard_text_values = None

        self.home_score = 0
        self.home_score_temp = 0

        self.home_team = None
        self.home_team_temp = 0
        self.home_team_fullname = None
        self.home_team_identified = False

        self.opponent_score = 0
        self.opponent_score_temp = 0

        self.opponent_team = None
        self.opponent_team_temp = None
        self.opponent_team_fullname = None
        self.opponent_team_identified = False

        self.match_time = None
        self.match_time_temp = None
        self._match_time_prev = []

        self.index = 0
        self.export_path = export_path
        self.filename_out = filename_out

    def analize_scoreboard(self):
        while self.index < ocr.video_length:
            try:
                ocr.eImageExported.wait()
                ocr.scoreboard.localize_scoreboard_image()
                ocr.scoreboard.split_scoreboard_image()
                ocr.scoreboard.enlarge_scoreboard_images(3)
                OCR_text = ocr.scoreboard.get_scoreboard_texts()
                ocr.football_match.provide_scoreboard_text_values(OCR_text)
                ocr.football_match.update_all_match_info()
                ocr.football_match.print_all_match_info()
                ocr.eImageExported.clear()
                self.index += 1
                ocr.index = self.index
            except Exception as e:
                logging.warning(e)

    def provide_scoreboard_text_values(self, scoreboard_text_values):

        self.scoreboard_text_values = scoreboard_text_values

    def cleanse_match_score(self):
        """
        Cleanse home_score_temp and opponent_score_temp values and removes
        noisy starters and enders if present

        :return: -
        """
        score_string = self.scoreboard_text_values[1].split(' ')[1]

        result = []
        for letter in score_string:
            if letter.isdigit():
                result += letter
        self.home_score_temp = result[0]
        self.opponent_score_temp = result[1]

    def cleanse_match_teams(self):
        """
        Cleanse home_team_temp and opponent_team_temp values and removes
        noisy starter or ender if present

        :return: -
        """
        self.home_team_temp = self.scoreboard_text_values[1].split(' ')[0]
        self.opponent_team_temp = self.scoreboard_text_values[1].split(' ')[2]

        # Check and remove noisy starters and enders
        if not self.home_team_temp[0].isalpha():
            self.home_team_temp = self.home_team_temp[1:4]
        elif not self.opponent_team_temp[-1].isalpha():
            self.opponent_team_temp = self.opponent_team_temp[0:3]

    def cleanse_match_time(self):
        """
        Cleanse match_time_temp, and removes noisy starter or ender if present

        :return: -
        """

        self.match_time_temp = self.scoreboard_text_values[0]

        # Check for noisy starters and ender and clean if present
        letter_ptr = 0
        if not self.match_time_temp[letter_ptr].isdigit():
            letter_ptr += 1
        if not self.match_time_temp[letter_ptr].isdigit():
            letter_ptr += 1
            self.match_time_temp = self.match_time_temp[letter_ptr:]
            logging.info("Time text noisy starter removed.")
        elif not self.match_time_temp[-1].isdigit():
            self.match_time_temp = self.match_time_temp[0:-1]
            logging.info("Time text noisy ender removed.")

    def update_match_time(self):
        """
        Validates cleansed match_time_temp with regular expression and sets match_time if valid value exists

        :return: True: time has been updated
                 False: time has not been updated
        """

        # Check if the OCR read value is valid
        time_expr = re.compile('\d\d:\d\d')
        res = time_expr.search(self.match_time_temp)

        if res is None:
            return False

        last_valid_timeval = self.match_time_temp[res.start():res.end()]
        self._match_time_prev.append(last_valid_timeval)

        # Check validity between last time values
        if last_valid_timeval < self._match_time_prev[len(self._match_time_prev)-2]:
            # Minute error occured - minute remain unchanged
            if last_valid_timeval[0:2] < self._match_time_prev[len(self._match_time_prev)-2][0:2]:
                logging.warning(
                    "Minute error occured: minute remain unchanged!")
                fixed_minutes = self._match_time_prev[len(
                    self._match_time_prev)-2][0:2]
                last_valid_timeval = fixed_minutes + last_valid_timeval[2:]

            else:
                # Second error occured - auto increment second
                logging.warning(
                    "Second error occured: auto incremented second!")
                seconds = self._match_time_prev[len(
                    self._match_time_prev)-2][-2:]
                fixed_seconds = str(int(seconds)+1)
                last_valid_timeval = last_valid_timeval[:-2] + fixed_seconds

        # Free unnecessary time values
        if len(self._match_time_prev) > 2:
            self._match_time_prev.pop(0)

        # Write all valid values to a text file for analysis
        self.match_time = last_valid_timeval
        with open(self.export_path + '/' + self.filename_out, 'a') as f:
            f.write("%s,%s\n" % (self.match_time, self.index))
        return True

    def update_match_score(self):
        """
        Validates cleansed score with regular expression

        :return: True: score matches the regexp
                 False: score does not match the regexp
        """
        score_expr = re.compile('\d-\d')
        res = score_expr.search(self.scoreboard_text_values[1])

        if res is None:
            return False

        self.home_score = self.home_score_temp
        self.opponent_score = self.opponent_score_temp
        return True

    def update_match_team(self, selected_team):
        """
        Sets cleansed home_team or opponent_team values if not set before

        :return: -
        """
        if selected_team == 'home':
            self.home_team = self.home_team_temp
            self.home_team_identified = True

        elif selected_team == 'opponent':
            self.opponent_team = self.opponent_team_temp
            self.opponent_team_identified = True

    def update_all_match_info(self):
        """
        Attempts to update match infos:
        time, teams, score
        :return: True: update succeed
                 False: update failed
        """
        if len(self.scoreboard_text_values[0]) > 0 and len(self.scoreboard_text_values[1]) > 0:
            try:
                # Clean OCR read time value and update time if valid
                self.cleanse_match_time()
                self.update_match_time()

                # Clean OCR read score value and update score if valid
                self.cleanse_match_score()
                self.update_match_score()

                # Clean OCR read team values and set teams if valid and necessary
                self.cleanse_match_teams()

                if self.home_team_identified is False:
                    self.update_match_team('home')

                if self.opponent_team_identified is False:
                    self.update_match_team('opponent')

            except Exception as e:
                logging.info(e)
                logging.info("Unable to update match info for some reason")
        else:
            logging.info("Unable to update match info: no text received!")

    def print_all_match_info(self):

        home_team_name = self.home_team
        opponent_team_name = self.opponent_team

        if self.home_team_fullname is not None and self.opponent_team_fullname is not None:
            home_team_name = self.home_team_fullname
            opponent_team_name = self.opponent_team_fullname

        print('{} {} {}-{} {}'.format(self.match_time,
                                      home_team_name,
                                      self.home_score,
                                      self.opponent_score,
                                      opponent_team_name))


# MAIN
# Empty times.txt file
def ocr(export_path, filename_in, filename_out, video_length):
    ocr.index = 0
    ocr.video_length = video_length
    open(export_path+'/' + filename_out, 'w').close()

    ocr.eImageExported = threading.Event()

        # Create objects and threads
    ocr.scoreboard = ImageHandler(export_path, filename_in)
    ocr.football_match = Match(export_path, filename_out)

    ocr.tImageExtractor = threading.Thread(
            None, ocr.scoreboard.extract_image_from_video, name="ImageExtractor")
    ocr.tScoreboardAnalyzer = threading.Thread(
            None, ocr.football_match.analize_scoreboard, name="ScoreboardAnalyzer")

    ocr.tImageExtractor.start()
    ocr.tScoreboardAnalyzer.start()

    ocr.tImageExtractor.join()
    ocr.tScoreboardAnalyzer.join()


if __name__ == '__main__' :
    filename_in = 'ocr/tmp/secondmatch.mkv'
    export_path = 'ocr/img'
    filename_out = 'times.txt'
    video_length = 1080
    ocr(export_path, filename_in, filename_out, 1080)