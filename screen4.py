import curses
import numpy as np
import cv2
import time
import threading
import os
import contextlib

import pygame
from pydub import AudioSegment
import ffmpeg as ffmpeg_lib
import tempfile
import sys
import random
import argparse
import mimetypes
import logging
import queue
from skimage.measure import block_reduce

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='The source is what is rendered. It can be a video or an image.')
    parser.add_argument('-a', '--ascii', action='store_true', help='Render using ascii characters')
    parser.add_argument('-d', '--dither', action='store_true', help='Render using dithered colors')
    parser.add_argument('-c', '--color', nargs='?', const=4, default=4, type=int, help='Render using color')
    parser.add_argument('-e', '--edgechar', nargs='?', default=None, const='█', type=str, help='Render edges')
    parser.add_argument('-l', '--lines', action='store_true', help='Render using ascii directional lines')
    parser.add_argument('-ec', '--edge-color', nargs=3, default=[-1, -1, -1], type=int, help='Color of the edges')


    return parser

def init_logger():
    log_file = 'screen4.log'
    if os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    #handler = logging.StreamHandler(sys.stdout)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

parser = init_parser()
logger = init_logger()
args = parser.parse_args()

def rgb_to_ansi_vectorized(frame):
    # we want to convert the whole frame to its corresponding ansi color code 
    frame = frame / 255
    frame = np.round(frame * 5)
    frame[:, :, 0] = frame[:, :, 0] * 36
    frame[:, :, 1] = frame[:, :, 1] * 6 
    frame = np.sum(frame, axis=2)
    frame = frame + 16
    frame = frame.astype(np.uint8)
    return frame

def to_ansi_color(frame, color_mode):

    # convert to rgb from bgr 

    # there will be 4 color modes
    # 1: black and white mode 
    # 2: grayscale mode
    # 3: 16 color mode 
    # 4: 256 color mode 
    # normalize the frame
    frame = frame / 255

    color_pallete_16 = [
            [0, 135, 81], # dark green
            [41, 173, 255], # blue
            [255, 119, 168], # pink
            [255, 204, 170], # light peach
            [41, 24, 16], # dark brown black
            [243, 239, 125], # light yellow
            #[190, 18, 80], # dark red
            [6, 90, 181], # true blue
            [255, 110, 89], # dark peach
            [35, 210, 140], # mint green
            [255, 240, 220], #pale white
            #[40, 20, 90], # dark blue
            [75, 20, 140], # dark purple
            [70, 30, 140], # dark purple
            [50,255,120], #  lighter green
            [46, 25, 29],
            [120, 240, 255], # light light blue
            [167, 20, 30], 
    ]



    ansi_standard_colors = color_pallete_16
    #ansi_standard_colors = pico8_palette

    

    ansi_colors = np.array(color_pallete_16)

    #normalize ansi 
    ansi_colors = ansi_colors / 255
    

    if color_mode == 0: 
        # just set everything to white 
        white_frame = np.ones_like(frame)
        frame = white_frame

    if color_mode == 1:
        # black and white mode
        frame = np.mean(frame, axis=2)
        frame = np.round(frame)
        frame = cv2.merge([frame, frame, frame])
    elif color_mode == 2:
        # ansi gray scale mode
        # 24 shades of gray 
        frame = np.mean(frame, axis=2)
        frame = np.round(frame * 23) / 23
        frame = cv2.merge([frame, frame, frame]).astype(np.float32)

    elif color_mode == 3:
        p = 0.8
        luminance = np.mean(frame, axis=(0, 1, 2))

        # if the luminance is below 0.5, then we squish the colors up
        if luminance < 0.5:
            frame = frame ** p / (frame ** p + (1 - frame) ** p)

        color_dists = np.linalg.norm(frame[..., np.newaxis, :] - ansi_colors, axis=3)
        closest_colors = np.argmin(color_dists, axis=2)
        frame = ansi_colors[closest_colors]



    elif color_mode == 4:
        # convert to 6 x 6 x 6 rgb color cube
        frame = np.round(frame * 5) / 5 

        
    frame = frame * 255
    frame = frame.astype(np.uint8)
    return frame 



def rgb_to_ansi_color_vectorized(frame):
    # we want to convert the whole frame to ansi color space

    n_frame = frame / 255

    quantized_frame = np.round(n_frame * 5) / 5

    quantized_frame = quantized_frame * 255

    return quantized_frame

def rgb_to_ansi_color_neighbours_vectorized(frame):

    n_frame = frame / 255

    quantized_frame = np.round(n_frame * 5) / 5

    other_corners = np.tile(np.expand_dims(quantized_frame, axis=2), (1, 1, 4, 1))

    rgb_scaled = n_frame * 5

    for i in range(3):
        int_val = np.floor(rgb_scaled[..., i])
        frac_val = rgb_scaled[..., i] - int_val
        other_corners[..., i + 1, i] = np.round(1 - frac_val + int_val) / 5

    other_corners = other_corners * 255

    return other_corners

def rgb_to_ansi_color_closest_triplet_vectorized(frame):

    neighbours = rgb_to_ansi_color_neighbours_vectorized(frame)

    # drop the neighbours that are the furthest away

    frame_tile = np.tile(np.expand_dims(frame, axis=2), (1, 1, 4, 1))

    dist = np.linalg.norm(neighbours - frame_tile, axis=3)

    # get the index of the max distance
    max_dist_index = np.argmax(dist, axis=2)

    # drop the furthest neighbour
    mask = np.arange(4).reshape(1, 1, 4) != max_dist_index[..., np.newaxis]

    filtered_neighbours = neighbours[mask].reshape(frame.shape[0], frame.shape[1], 3, 3)

    # remove the argmax index from  neighbours
    # shape neighbours: (H, W, 4, 3)
    # shape max_dist_index: (H, W)
    # shape closest_neighbours: (H, W, 3, 3)
    return filtered_neighbours


def weighted_nearest_neighbour_vectorized(p, neighbours):
    # we want to tile p such that we can subtract it from each of its 4 neighbours
    # so it needs to be tile 4 times on the axis 2
    # p is the frame whichi is (H, W, 3)
    # neighbours is (H, W, 4, 3)
    p_tile = np.tile(np.expand_dims(p, axis=2), (1, 1, neighbours.shape[2], 1))
    weights = neighbours - p_tile
    weights = np.linalg.norm(weights, axis=3)
    total_distance = np.sum(weights, axis=2)
    weights = weights / total_distance[..., np.newaxis]
    return weights

def get_dither_thresholds(k):
    # use the recursive method to generate the dither thresholds
    # get nearest power of 2
    n = 2**int(np.ceil(np.log2(k)))

    if n == 1:
        return np.array([[0]])
    else:
        smaller = get_dither_thresholds(n//2)
        tiles = [ 4 * smaller + i for i in range(4)]
        top = np.hstack((tiles[0], tiles[1]))
        bottom = np.hstack((tiles[2], tiles[3]))
        return np.vstack((top, bottom))

def multicolored_dither_choice_vectorized(k, choices, weights):
    bayer_matrix = (get_dither_thresholds(k) + 0.5) / k**2

    # tile the bayer matrix to the size of the frame , but trim the excess
    H, W = choices.shape[:2]
    n_h, n_w = H / k, W / k
    n_h, n_w = int(np.ceil(n_h)), int(np.ceil(n_w))
    tile_bayer = np.tile(bayer_matrix, (n_h, n_w))
    tile_bayer = tile_bayer[:H, :W]

    # we want to get the chosen index for each pixel
    cum_weights = np.cumsum(weights, axis=2)
    # now we want to select for indices whos cum sum val is greater than the bayer matrix value
    # this is ordered dithering so no random
    chosen_indices = (cum_weights > tile_bayer[..., np.newaxis]).argmax(axis=-1)
    # now we want to get the chosen color for each pixel
    # we want to get the chosen color for each pixel
    chosen_colors = choices[np.arange(H)[:, np.newaxis], np.arange(W), chosen_indices]
    return chosen_colors

def ordered_color_dither(frame):
    new_frame = np.zeros_like(frame)

    #neighbours = rgb_to_ansi_color_neighbours_vectorized(frame)
    neighbours = rgb_to_ansi_color_closest_triplet_vectorized(frame)

    weights = weighted_nearest_neighbour_vectorized(frame, neighbours)
    shortest_side = min(frame.shape[0], frame.shape[1])
    n = int(shortest_side * 0.025)

    # be careful of overflow error 
    if n < 2:
        k = 2
    else: 
        k = 2**int(np.ceil(np.log2(n)))

    k = max(2, k)

    bayer_matrix = (get_dither_thresholds(k) + 1) / k**2


    new_frame = multicolored_dither_choice_vectorized(k, neighbours, weights)

    new_frame_undithered = rgb_to_ansi_color_vectorized(frame)
    # we want an error threshold that triggers dithering, otherwise we just use the closest color
    error = frame - new_frame_undithered
    error = np.linalg.norm(error, axis=2)

    dither_threshold = 44.167 * 0.6
    new_frame[error < dither_threshold] = new_frame_undithered[error < dither_threshold]
    return new_frame



class AudioPlayer(threading.Thread):

    def __init__(self, events):
        super().__init__()
        self.events = events
        self.stop_flag = events['stop_flag']
        self.daemon = True

    def run(self):
        logger.info('AudioPlayerActive')
        audio = AudioSegment.from_file(args.source)
        audio.export('temp.wav', format='wav')
        pygame.mixer.init()
        pygame.mixer.music.load('temp.wav')
        pygame.mixer.music.play()
        while not self.stop_flag.is_set():
            if not pygame.mixer.music.get_busy():
                break
        pygame.mixer.quit()
        logger.info('Stopping AudioPlayer')











class TerminalScreen(threading.Thread):

    def __init__(self, events, stdscr):
        super().__init__()
        self.input_queue = queue.Queue()
        self.stdscr = stdscr
        self.events = events
        self.stop_flag = events['stop_flag']
        self.daemon = True
        self.prev_dims = (0, 0)

    def handle_inputs(self):
        key = self.stdscr.getch()
        if key == ord('q'):
            self.stop_flag.set()
            return
        if key == ord(' '):
            pause = self.events['pause']
            if pause.is_set():
                pause.clear()
            else:
                pause.set()


    def get_dimensions(self, image):
        # we want to get the appropriate dimensions to display the image
        # inside the terminal, such that the image is scaled to fit
        # the terminal window, while maintaining the aspect ratio
        # are "pixels" are 2 x 1 so we need make sure the width is double what it should be
        terminal_height, terminal_width = self.stdscr.getmaxyx()
        image_height, image_width, _ = image.shape
        aspect_ration = image_width / image_height
        target_width = terminal_width
        target_height = int(target_width / aspect_ration * 0.5)
        if target_height > terminal_height:
            target_height = terminal_height
            target_width = int(target_height * aspect_ration * 2)
        return target_height, target_width
    
    def chr_to_ansi(self, char, ansi_code):
        return f'\033[{ansi_code}m{char}\033[0m'

    def rgb_to_hsv(self, r, g, b):
        r, g, b = r / 255, g / 255, b / 255
        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin

        if delta == 0:
            h = 0
        elif cmax == r:
            h = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            h = 60 * (((b - r) / delta) + 2)
        elif cmax == b:
            h = 60 * (((r - g) / delta) + 4)

        if cmax == 0:
            s = 0
        else:
            s = delta / cmax

        v = cmax

        return h, s, v

    def get_render_char(self, x, y, image, edges, grad, luminance, dither_mask):
        gradient_chars = "█●●ØBB@@&&WM#Æ%hkbqOQCJUYX{{{?+++!!!░▒█"
        if args.color == 3:
            gradient_chars = gradient_chars[:-3]
        gradient_chars = gradient_chars[::-1]
        edge_map = {
                (255, 0, 0): '|',
                (255, 255, 0): '/',
                (0, 255, 0): '-',
                (0, 0, 255): '\\'
        }

        char = '█'
        r, g, b = image[y, x]
        edge_color = edges[y, x]
        grad_color = grad[y, x]
        if args.dither:
            if dither_mask[y, x]:
                char = ':'
        if args.ascii:
            gradient_index = int(luminance[y, x] * (len(gradient_chars)))
            gradient_index = min(gradient_index, len(gradient_chars) - 1)
            char = gradient_chars[gradient_index]
        
        if args.dither:
            if not dither_mask[y, x]:
                char = '.'

        if args.edgechar != None:
            if edge_color[0] == 255:
                char = args.edgechar 

        if args.lines:
            char = edge_map.get(tuple(grad_color), char)

        return char



    def render(self, image, frame_number, total_frames):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        target_height, target_width = self.get_dimensions(image)
        terminal_height, terminal_width = self.stdscr.getmaxyx()
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)



        if self.prev_dims != (target_height, target_width):
            self.stdscr.clear()
            self.prev_dims = (target_height, target_width)

         # get per_pixel_luminance
        luminance = np.mean(image, axis=2)
        luminance = luminance / 255


        edges, grad = self.get_edges(image, (target_width, target_height))

        colored_image = to_ansi_color(image, args.color)
        dither_mask = np.zeros_like(luminance)
        if args.dither: 
            k = 4 # must be a power of 2 
            # get the mean luminance of the image as a single value 
            mean_luminance = np.mean(luminance)
            #if mean_luminance > 0.5:
            #    luminance = luminance ** 2.7
            dither_luminance = luminance ** 2.7
            dither_thresholds = get_dither_thresholds(4)
            dither_thresholds = (dither_thresholds + 0.5) / k**2

            # get image saturation values 
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)



            # get dither mask 
            # tile get the index for the dither threshold matrix at each position by taking the positions modulo 
            rows, cols = np.indices(dither_mask.shape)
            dither_mask = dither_thresholds[rows % k, cols % k]
            # compare to luminance to create the mask 
            dither_mask = dither_luminance > dither_mask

            # in all places where saturation is above 0.5 and luminance is above 0.5
            # we set dither mask to 1 
            dither_bright_color_mask = (hsv_image[..., 1] > 50) & (dither_luminance > 0.2) 
            dither_mask[dither_bright_color_mask] = 1

            # set the color to black if the dither mask is false 
            colored_image[~dither_mask] = [0, 0, 0]

        #draw edge color 
        if (args.edgechar or args.lines) and -1 not in args.edge_color:
            edge_color = np.array(args.edge_color)
            edge_color = np.tile(edge_color, (target_height, target_width, 1))
            colored_image = np.where(edges == 255, edge_color, colored_image)



        ansi_codes = rgb_to_ansi_vectorized(colored_image)
        

        edge_map = {
                (255, 0, 0): '|',
                (255, 255, 0): '/',
                (0, 255, 0): '-',
                (0, 0, 255): '\\'
        }

        for y in range(target_height-1):
            for x in range(target_width-1):
                if y >= terminal_height or x >= terminal_width:
                    continue
                
                # switches:
                # ascii 
                # dither 
                # color : all white, grayscale, 16 color, 256 color

                # we want to handle a couple different combinations of cases 
                char = self.get_render_char(x, y, image, edges, grad, luminance, dither_mask)
                curse_color = curses.color_pair(ansi_codes[y, x])
                self.stdscr.addstr(y, x, char, curse_color)

        fill_char = '█'    
        progress = frame_number / total_frames
        progress = int(progress * target_width)
        for x in range(progress):
            self.stdscr.addstr(target_height-1, x, fill_char, curses.color_pair(196))

        self.stdscr.refresh()

    def get_two_way_edges(self, image, dimensions):
        # we want to be able to get a map of the inside of an edge, and an outside of an edge
        # resize image
        image = cv2.resize(image, dimensions)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        canny = cv2.Canny(gray, 100, 200)


        # dilate the edges
        kernel = np.ones((3,3), np.uint8)
        external_edges = cv2.dilate(canny, kernel, iterations=1)

        # erode the edges for the internal boundary
        internal_edges = cv2.erode(canny, kernel, iterations=1)

        inside_mask = np.zeros_like(image)
        inside_mask[internal_edges > 0] = [255, 0, 0]

        outside_mask = np.zeros_like(image)
        outside_mask[external_edges > 0] = [255, 255, 255]

        edge_colors = np.zeros_like(image)
        # add the inside and outside masks together
        edge_colors = outside_mask

        return edge_colors



    def get_edges(self, image, dimensions):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # difference of guassians for edge detection
        blur1 = cv2.GaussianBlur(gray, (13,13), 5.0)
        blur2 = cv2.GaussianBlur(gray, (7,7), 1.0)

        tau = 0.5
        blur1 = cv2.multiply(blur1, 1-tau)
        blur2 = cv2.multiply(blur2, tau)
        
        dog = cv2.subtract(blur1,blur2)

        dog[dog>4] = 255

        edges = cv2.resize(dog, dimensions)
        resize_image = cv2.resize(image, dimensions)

        edges[edges>15] = 255

        
        edges = cv2.Canny(resize_image, 150, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


        # apply directional sobel filter
        # we want the image gradient in the x and y directional

        # apply resize gray to dimensions
        gray = cv2.resize(gray, dimensions)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)


        # get the angle of the gradient
        angle = np.arctan2(sobel_y, sobel_x)
        #map from -pi to pi to 0 to 1
        angle = (angle / np.pi + 1) / 2

        # add we want our 

        # quantize into 8 different values, 0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8

        angle += 1/16

        angle = angle % 1

        angle = np.floor(angle * 8) / 8

        color_map = {
            0: [255, 0, 0],
            1/8: [255, 255, 0],
            2/8: [0, 255, 0],
            3/8: [0, 0, 255],
            4/8: [255, 0, 0],
            5/8: [255, 255, 0],
            6/8: [0, 255, 0],
            7/8: [0,0,255]
        }

        grad = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

        for i in range(8):
            mask = angle == i/8
            grad[mask] = color_map[i/8]


        mask = cv2.cvtColor(edges, cv2.COLOR_RGB2GRAY)
        # make it a binary mask, where if color is above 100, it is 1, else 0
        _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

        mask = mask.astype(np.float32)

        # multiply by 255
        mask = mask / 255

        mask3 = cv2.merge([mask, mask, mask])

        grad = cv2.multiply(grad.astype(np.float32), mask3, dtype=cv2.CV_32F)

        #edges = cv2.resize(edges, dimensions)



        return edges, grad

    
    def get_dither_thresholds(self, k):
        # use the recursive method to generate the dither thresholds
        # get nearest power of 2
        n = 2**int(np.ceil(np.log2(k)))

        if n == 1:
            return np.array([[0]])
        else:
            smaller = self.get_dither_thresholds(n//2)
            tiles = [ 4 * smaller + i for i in range(4)]
            top = np.hstack((tiles[0], tiles[1]))
            bottom = np.hstack((tiles[2], tiles[3]))
            return np.vstack((top, bottom))


    def run(self):
        while not self.stop_flag.is_set():
            logger.info("TerminalScreenActive")

            self.handle_inputs()
            try:
                f_num, t_frames, image = self.input_queue.get(timeout=1)

                # log that the image was received
                logger.info(f'Image received: {image.shape}')
                

                self.render(image, f_num, t_frames)

                self.input_queue.task_done()
            except queue.Empty:
                continue

        logger.info('Stopping Terminal Screen')

import ffmpeg


class WebCamPlayer(threading.Thread):
    
    def __init__(self, events, screen_input_queue):
        super().__init__()
        self.events = events
        self.stop_flag = events['stop_flag']
        self.screen_input_queue = screen_input_queue
        self.daemon = True

    def run(self):
        logger.info('WebCamPlayerActive')

        spf = 1.0 / 30.0
        prev_frame_time = time.time()
        try:
            webcam = cv2.VideoCapture(0)
        except Exception as e:
            logger.error(f'Error opening webcam: {e}')
            return
        while not self.stop_flag.is_set():
            logger.info('WebCamPlayerActive')
            elapsed_time = time.time() - prev_frame_time
            if elapsed_time < spf:
                continue
            prev_frame_time = time.time()
            ret, frame = webcam.read()
            if not ret:
                break

            # flip the image left to right
            frame = cv2.flip(frame, 1)
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # if there is anything on the queue, skip this frame
            if not self.screen_input_queue.empty():
                continue
            self.screen_input_queue.put((0, 1, frame))
        logger.info('Stopping WebCamPlayer')

class VideoPlayer(threading.Thread):

    def __init__(self, events, screen_input_queue):
        super().__init__()
        self.events = events
        self.stop_flag = events['stop_flag']
        self.screen_input_queue = screen_input_queue
        self.daemon = True

    def run(self):
        logger.info('VideoPlayerActive')
        video = cv2.VideoCapture(args.source)
        fps = video.get(cv2.CAP_PROP_FPS)
        spf = 1 / fps
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        prev_time = time.time()

        cur_frame = 0
        while not self.stop_flag.is_set():


            # check if the other frame is done and the queue is Empty
            #if not self.screen_input_queue.empty():
            #    continue
            elapsed_time = time.time() - start_time
            frame_number = int(elapsed_time * fps)

            logger.info(f'FrameNumber: {frame_number}')

            while cur_frame <= frame_number:
                ret, frame = video.read()
                if not ret:
                    break
                cur_frame += 1

            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # if there is anything on the queue, skip this frame
            if not self.screen_input_queue.empty():
                continue

            if not ret:
                break

            self.screen_input_queue.put((cur_frame,total_frames,frame))
            time.sleep(spf)

        video.release()

            
        logger.info('Stopping VideoPlayer')

class ImagePlayer(threading.Thread):

    def __init__(self, events, screen_input_queue):
        super().__init__()
        self.events = events
        self.stop_flag = events['stop_flag']
        self.screen_input_queue = screen_input_queue
        self.daemon = True

    def run(self):
        image = cv2.imread(args.source)
        while not self.stop_flag.is_set():
            img_cpy = np.array(image)
            self.screen_input_queue.put((0, 1, img_cpy))
            time.sleep(0.2)
        logger.info('Stopping ImagePlayer')

def curses_main(stdscr, events):
    events['pause'] = threading.Event()

    terminal_screen = TerminalScreen(events, stdscr)

    terminal_screen.start()

    if args.source == 'webcam':
        player = WebCamPlayer(events, terminal_screen.input_queue)
        player.start()

        try:
            events['stop_flag'].wait()
        except KeyboardInterrupt:
            events['stop_flag'].set()
        finally:
            terminal_screen.join()
            player.join()
        return

    
    media_type, _ = mimetypes.guess_type(args.source)

    if media_type.startswith('video'):
        player = VideoPlayer(events, terminal_screen.input_queue)
        audio_player = AudioPlayer(events)
    elif media_type.startswith('image'):
        player = ImagePlayer(events, terminal_screen.input_queue)
    else:
        logger.error(f'Unsupported media type: {media_type}')
        return
    

    global start_time
    start_time = time.time()
    player.start()
    if media_type.startswith('video'):
        audio_player.start()



    
    try:
        events['stop_flag'].wait()
    except KeyboardInterrupt:
        events['stop_flag'].set()
    finally:
        terminal_screen.join()
        player.join()
        if media_type.startswith('video'):
            audio_player.join()


def init_curses():
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    curses.curs_set(0)
    stdscr.keypad(True)
    curses.start_color()
    stdscr.timeout(1)
    for i in range(1, 256):
        if args.ascii or args.dither:
            curses.init_pair(i, i, 0)
        else:
            curses.init_pair(i, i, i)

    curses.init_pair(1, 1, 0)
    return stdscr


start_time = time.time()
def main():    
    logger.info(f'ASCII_MODE:{args.ascii}')
    logger.info(f'SOURCE:{args.source}')

    media_type, _ = mimetypes.guess_type(args.source)
    if not os.path.exists(args.source) and args.source != 'webcam':
        logger.error(f'File does not exist: {args.source}')
        return

    events = { 'stop_flag': threading.Event() }
    def _curses_main(stdscr):
        return curses_main(stdscr, events)
    
    init_curses()
    
    try:
        curses.wrapper(_curses_main)
    except Exception as e:
        print(f"Error: {e}")
        events['stop_flag'].set()


if __name__ == '__main__':
    main()
