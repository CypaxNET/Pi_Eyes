#!/usr/bin/env python

# This is a modified version of cyclops.py (designed for the Gakken WorldEye display).
# It uses the Raspberry PI camera togther with OpenCV to focus the eye on any detected motion.
# Code is still in-progress and clud need some clean-up.

import pygame

import math
import pi3d
import random
import thread
import time
from svg.path import Path, parse_path
from xml.dom.minidom import parse
from gfxutil import *

import numpy as np
import sys
import os
import subprocess
import glob

from picamera.array import PiRGBArray
from picamera import PiCamera

import argparse
import warnings
import datetime
import imutils
import json				# required for config file
import cv2				# OpenCV


def print_there(x, y, text):
	sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (y, x, text))
	sys.stdout.flush()


def cvThread():

	global avg
	global isMotionDetected
	global motionX
	global motionY
	global imgBrightness
	global picam
	global sound_FILE_PATTERN
	global cv_DETECT_MOVEMENT
	global picam_RawCapture
	global doPlaySound
	
	prevIsMotionDetected = isMotionDetected

	
	nextTimeExposureAdjustStart = time.time() # when to perform next exposure adjustment
	nextTimeExposureAdjustEnd = 0.0
	
	timeBefore = time.time()
	framesBefore = 0
	frames = 0

	# capture frames from the camera
	for f in picam.capture_continuous(picam_RawCapture, format="bgr", use_video_port=True):
		frames += 1
		
		# grab the raw NumPy array representing the image and initialize
		# the timestamp and occupied/unoccupied text
		frame = f.array
		foundMovement = False

		# every once in a time adjust to the environment brightness
		if ((time.time() >= nextTimeExposureAdjustStart) and (imgBrightness < 0.6) ):
			picam.exposure_mode = 'auto'
			nextTimeExposureAdjustStart = time.time() + 300.0
			nextTimeExposureAdjustEnd = time.time() + 5.0
			print_there(1, 27, "exposure mode: auto    ")
		
		if (time.time() >= nextTimeExposureAdjustEnd and picam.exposure_mode == 'auto'):
			picam.exposure_mode = 'off'
			print_there(1, 27, "exposure mode: off    ")
		
		# resize the frame, convert it to grayscale, and blur it
		frame = imutils.resize(frame, width=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21, 21), 0)
		
		# calculate image brightness
		# it appears to me, that cv2.mean(gray)[0] / 255 is slightly faster than np.average(gray) / 255
		imgBrightness = cv2.mean(gray)[0] / 255
		#imgBrightness = np.average(gray) / 255
		
		print_there(1, 28, "brightness: %.1f%%     " % (imgBrightness*100.0))

		# if the average frame is None, initialize it
		if avg is None:
			print("[INFO] starting background model...")
			avg = gray.copy().astype("float")
			picam_RawCapture.truncate(0)
			continue

		# accumulate the weighted average between the current frame and
		# previous frames, then compute the difference between the current
		# frame and running average
		cv2.accumulateWeighted(gray, avg, 0.5)
		
		if (cv_DETECT_MOVEMENT == True):
			frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
			
			# threshold the delta image, dilate the thresholded image to fill
			# in holes, then find contours on thresholded image
			thresh = cv2.threshold(frameDelta, conf["picam_DELTA_TRESH"], 255, cv2.THRESH_BINARY)[1]
			thresh = cv2.dilate(thresh, None, iterations=2)
			cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cnts = cnts[0] if imutils.is_cv2() else cnts[1]
			
			# loop over the contours
			biggestRect = 0
			biggestRectX = -1
			biggestRectY = -1
			for c in cnts:
				# if the contour is too small, ignore it
				if cv2.contourArea(c) < conf["picam_MIN_AREA"]:
					continue
				
				# compute the bounding box for the contour, draw it on the frame,
				# and update the text
				foundMovement = True
				(x, y, w, h) = cv2.boundingRect(c)
				if ((w*h) > biggestRect):
					biggestRectX = x + w/2
					biggestRectY = y + h/2
					#cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
				
			if foundMovement:
				motionX = biggestRectX
				motionY = biggestRectY
				isMotionDetected = True
				print_there(1, 29, "motion detected at %d, %d        " % ( motionX, motionY ) )
			else:
				isMotionDetected = False
				print_there(1, 29, "no motion detected               ")
			
		# play sound file on movement
		if ((isMotionDetected == True) and (prevIsMotionDetected == False)):
			doPlaySound = True
			
		# clear the stream in preparation for the next frame
		picam_RawCapture.truncate(0)
		
		prevIsMotionDetected = isMotionDetected
		
		now = time.time()
		if(now > (timeBefore + 1.0)):
			print_there(1, 25, "OpenCV FPS: %.1f      " % ((frames-framesBefore)/(now-timeBefore)) )
			framesBefore = frames
			timeBefore = now


# Generate one frame of imagery
def frame(p):
	global startX, startY, destX, destY, curX, curY
	global moveDuration, holdDuration, startTime, isMoving
	global frames
	global irisMesh
	global eye_PupilMinPoints, eye_PupilMaxPoints, eye_IrisPoints, irisZ
	global eyeLathe
	global upperEyelidMesh, lowerEyelidMesh
	global eye_UpperLidOpenPoints, eye_UpperLidClosedPoints, eye_LowerLidOpenPoints, eye_LowerLidClosedPoints
	global eye_UpperLidEdgePoints, eye_LowerLidEdgePoints
	global prevUpperLidPts, prevLowerLidPts
	global prevUpperLidWeight, prevLowerLidWeight
	global prevPupilScale
	global irisRegenThreshold, upperLidRegenThreshold, lowerLidRegenThreshold
	global luRegen, llRegen, ruRegen, rlRegen
	global timeOfLastBlink, timeToNextBlink
	global blinkState
	global blinkDuration
	global blinkStartTime
	global trackingPos
	
	global motionX
	global motionY
	global isMotionDetected
	
	global eye_ANIMATED_IRIS
	global irisAnimationFrames
	global num_IrisAnimationFrames
	
	global framesBefore
	global fpsTime
	
	DISPLAY.loop_running()
	
	now = time.time()
	dt  = now - startTime
	
	frames += 1
	
	if(now > (fpsTime + 1.0)):
		print_there(1, 23, "Render FPS: %.1f      " % ((frames-framesBefore)/(now-fpsTime)) )
	
	print_there(1, 18, "isMoving: %d               " % ( isMoving ) )
	
	# Eye position
	if (isMotionDetected):
		if dt <= moveDuration:
			scale        = (now - startTime) / moveDuration
			# Ease in/out curve: 3*t^2-2*t^3
			scale = 3.0 * scale * scale - 2.0 * scale * scale * scale
			curX         = startX + (destX - startX) * scale
			curY         = startY + (destY - startY) * scale
		else:
			destX        = ((500 - motionX) / 500.0) * 60.0 - 30.0
			n            = math.sqrt(900.0 - destX * destX)
			destY        = 0
		
			deltaX = abs(curX - destX)
			if deltaX > 3.0:
				if curX > destX:
					curX -= 1.0
				else:
					curX += 1.0
			else:
				curX = destX
			
			deltaY = abs(curY - destY)
			if deltaY > 3.0:
				if curY > destY:
					curY -= 1.0
				else:
					curY += 1.0
			else:
				curY = destY
			
			if (deltaX <= 3.0 and deltaY <= 3.0):
				holdDuration = random.uniform(4.0, 6.5)
				moveDuration = random.uniform(0.22, 0.55)
				startTime    = now
				isMoving     = False
				startX       = destX
				startY       = destY
		
	else:
		if isMoving == True:
			if dt <= moveDuration:
				scale        = (now - startTime) / moveDuration
				# Ease in/out curve: 3*t^2-2*t^3
				scale = 3.0 * scale * scale - 2.0 * scale * scale * scale
				curX         = startX + (destX - startX) * scale
				curY         = startY + (destY - startY) * scale
			else:
				startX       = destX
				startY       = destY
				curX         = destX
				curY         = destY
				holdDuration = random.uniform(0.9, 2.7)
				startTime    = now
				isMoving     = False
		else:
			if dt >= holdDuration:
				destX        = random.uniform(-30.0, 30.0)
				n            = math.sqrt(900.0 - destX * destX)
				destY        = random.uniform(-n, n)
				# Movement is slower in this version because
				# the WorldEye display is big and the eye
				# should have some 'mass' to it.
				moveDuration = random.uniform(0.22, 0.55)
				startTime    = now
				isMoving     = True

	print_there(1, 10, "startX: %d   startY: %d                       " % ( startX, startY ) )
	print_there(1, 11, "destX: %d    destY: %d                        " % ( destX, destY ) )


	# Regenerate iris geometry only if size changed by >= 1/2 pixel
	if abs(p - prevPupilScale) >= irisRegenThreshold:
		# Interpolate points between min and max pupil sizes
		interPupil = pointsInterp(eye_PupilMinPoints, eye_PupilMaxPoints, p)
		# Generate mesh between interpolated pupil and iris bounds
		mesh = pointsMesh(None, interPupil, eye_IrisPoints, 4, -irisZ, True)
		irisMesh.re_init(pts=mesh)
		prevPupilScale = p

	# Eyelid WIP

	if eye_AUTOBLINK and (now - timeOfLastBlink) >= timeToNextBlink:
		# Similar to movement, eye blinks are slower in this version
		timeOfLastBlink = now
		duration        = random.uniform(0.06, 0.12)
		if blinkState != 1:
			blinkState     = 1 # ENBLINK
			blinkStartTime = now
			blinkDuration  = duration
		timeToNextBlink = duration * 3 + random.uniform(0.0, 8.0)

	if blinkState: # Eye currently winking/blinking?
		# Check if blink time has elapsed...
		if (now - blinkStartTime) >= blinkDuration:
			blinkState += 1
			if blinkState > 2:
				blinkState = 0 # NOBLINK
			else:
				blinkDuration *= 2.0
				blinkStartTime = now

	if eye_LID_TRACKS_PUPIL:
		# 0 = fully up, 1 = fully down
		n = 0.5 - curY / 70.0
		if   n < 0.0: n = 0.0
		elif n > 1.0: n = 1.0
		trackingPos = (trackingPos * 3.0 + n) * 0.25

	if blinkState:
		n = (now - blinkStartTime) / blinkDuration
		if n > 1.0: n = 1.0
		if blinkState == 2: n = 1.0 - n
	else:
		n = 0.0
	newUpperLidWeight = trackingPos + (n * (1.0 - trackingPos))
	newLowerLidWeight = (1.0 - trackingPos) + (n * trackingPos)

	if (ruRegen or (abs(newUpperLidWeight - prevUpperLidWeight) >=
		upperLidRegenThreshold)):
		newUpperLidPts = pointsInterp(eye_UpperLidOpenPoints,
		eye_UpperLidClosedPoints, newUpperLidWeight)
		if newUpperLidWeight > prevUpperLidWeight:
			upperEyelidMesh.re_init(pts=pointsMesh(
			eye_UpperLidEdgePoints, prevUpperLidPts,
			newUpperLidPts, 5, 0, False, True))
		else:
			upperEyelidMesh.re_init(pts=pointsMesh(
			eye_UpperLidEdgePoints, newUpperLidPts,
			prevUpperLidPts, 5, 0, False, True))
		prevUpperLidWeight = newUpperLidWeight
		prevUpperLidPts    = newUpperLidPts
		ruRegen = True
	else:
		ruRegen = False

	if (rlRegen or (abs(newLowerLidWeight - prevLowerLidWeight) >=
		lowerLidRegenThreshold)):
		newLowerLidPts = pointsInterp(eye_LowerLidOpenPoints,
		eye_LowerLidClosedPoints, newLowerLidWeight)
		if newLowerLidWeight > prevLowerLidWeight:
			lowerEyelidMesh.re_init(pts=pointsMesh(
			eye_LowerLidEdgePoints, prevLowerLidPts,
			newLowerLidPts, 5, 0, False, True))
		else:
			lowerEyelidMesh.re_init(pts=pointsMesh(
			eye_LowerLidEdgePoints, newLowerLidPts,
			prevLowerLidPts, 5, 0, False, True))
		prevLowerLidWeight = newLowerLidWeight
		prevLowerLidPts    = newLowerLidPts
		rlRegen = True
	else:
		rlRegen = False

	# Draw eye

	irisMesh.rotateToX(curY)
	irisMesh.rotateToY(curX)
	
	if eye_ANIMATED_IRIS:
		i_n = frames % num_IrisAnimationFrames
		irisMesh.set_textures([irisAnimationFrames[i_n]])
		print_there(1, 19, "iris image # %d         " % ( i_n ) )
	
	irisMesh.draw()
	
	eyeLathe.rotateToX(curY)
	eyeLathe.rotateToY(curX)
	eyeLathe.draw()
	upperEyelidMesh.draw()
	lowerEyelidMesh.draw()
	
	# clear terminal output to get rid off the glGetError 0x500 warnings caused by irisMesh.set_textures
	if (frames == (num_IrisAnimationFrames+1)):
		print(chr(27) + "[2J")

#clear screen
print(chr(27) + "[2J")

pygame.init()

# construct the argument parser and parse the arguments
thisArgumentParser = argparse.ArgumentParser()
thisArgumentParser.add_argument("-c", "--conf", required = True, help = "path to the JSON configuration file")
args = vars(thisArgumentParser.parse_args())

# filter warnings and load the configuration file
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))

# initialize the camera and grab a reference to the raw camera capture
picam = PiCamera()
picam.resolution = tuple(conf["picam_RESOLUTION"])
picam.framerate = conf["picam_FPS"]
picam_RawCapture = PiRGBArray(picam, size=tuple(conf["picam_RESOLUTION"]))

cv_DETECT_MOVEMENT = conf["cv_DETECT_MOVEMENT"]
sound_FILE_PATTERN = conf["sound_FILE_PATTERN"]

# CONFIG for eye motion ----------------------------------------------
eye_PUPIL_IN_FLIP     = conf["eye_PUPIL_IN_FLIP"]    # If True, reverse reading from PUPIL_IN
eye_LID_TRACKS_PUPIL  = conf["eye_LID_TRACKS_PUPIL"] # If True, eyelid tracks pupil
eye_PUPIL_MIN         = conf["eye_PUPIL_MIN"]        # Lower analog range from PUPIL_IN
eye_PUPIL_MAX         = conf["eye_PUPIL_MAX"]        # Upper "
eye_AUTOBLINK         = conf["eye_AUTOBLINK"]        # If True, eye blinks autonomously

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["picam_WARMUP_TIME"])
avg = None

isMotionDetected = False
motionX = 0
motionY = 0
imgBrightness = 0.0

doPlaySound = False

lastTimeSound = 0
lastSoundFile = ["", "", ""]

# Load SVG file, extract paths & convert to point lists --------------------

# Thanks Glen Akins for the symmetrical-lidded cyclops eye SVG!
# Iris & pupil have been scaled down slightly in this version to compensate
# for how the WorldEye distorts things...looks OK on WorldEye now but might
# seem small and silly if used with the regular OLED/TFT code.
eye_DOM                  = parse(conf["eye_DOM"])
eye_ViewBox              = getViewBox(eye_DOM)
eye_PupilMinPoints       = getPoints(eye_DOM, "pupilMin"      , 32, True , True )
eye_PupilMaxPoints       = getPoints(eye_DOM, "pupilMax"      , 32, True , True )
eye_IrisPoints           = getPoints(eye_DOM, "iris"          , 32, True , True )
eye_ScleraFrontPoints    = getPoints(eye_DOM, "scleraFront"   ,  0, False, False)
eye_ScleraBackPoints     = getPoints(eye_DOM, "scleraBack"    ,  0, False, False)
eye_UpperLidClosedPoints = getPoints(eye_DOM, "upperLidClosed", 33, False, True )
eye_UpperLidOpenPoints   = getPoints(eye_DOM, "upperLidOpen"  , 33, False, True )
eye_UpperLidEdgePoints   = getPoints(eye_DOM, "upperLidEdge"  , 33, False, False)
eye_LowerLidClosedPoints = getPoints(eye_DOM, "lowerLidClosed", 33, False, False)
eye_LowerLidOpenPoints   = getPoints(eye_DOM, "lowerLidOpen"  , 33, False, False)
eye_LowerLidEdgePoints   = getPoints(eye_DOM, "lowerLidEdge"  , 33, False, False)


# Set up display and initialize pi3d ---------------------------------------

DISPLAY = pi3d.Display.create(samples=4)
DISPLAY.set_background(0, 0, 0, 1) # r,g,b,alpha

# eye_Radius is the size, in pixels, at which the whole eye will be rendered.
if DISPLAY.width <= (DISPLAY.height * 2):
	# For WorldEye, eye size is - almost- full screen height
	eye_Radius   = DISPLAY.height / 2.1
else:
	eye_Radius   = DISPLAY.height * 2 / 5

# A 2D camera is used, mostly to allow for pixel-accurate eye placement,
# but also because perspective isn't really helpful or needed here, and
# also this allows eyelids to be handled somewhat easily as 2D planes.
# Line of sight is down Z axis, allowing conventional X/Y cartesion
# coords for 2D positions.
cam    = pi3d.Camera(is_3d=False, at=(0,0,0), eye=(0,0,-1000))
shader = pi3d.Shader("uv_light")
light  = pi3d.Light(lightpos=(0, -500, -500), lightamb=(0.2, 0.2, 0.2))


# Load texture maps --------------------------------------------------------
eye_IRIS_MAP   = pi3d.Texture(conf["eye_IRIS_MAP"],   mipmap=False, filter=pi3d.GL_LINEAR)
eye_SCLERA_MAP = pi3d.Texture(conf["eye_SCLERA_MAP"], mipmap=False, filter=pi3d.GL_LINEAR, blend=True)
eye_LID_MAP    = pi3d.Texture(conf["eye_LID_MAP"] ,   mipmap=False, filter=pi3d.GL_LINEAR, blend=True)
# U/V map may be useful for debugging texture placement; not normally used
#uvMap     = pi3d.Texture("graphics/uv.png"    , mipmap=False,
#              filter=pi3d.GL_LINEAR, blend=False, m_repeat=True)


# Initialize static geometry -----------------------------------------------

# Transform point lists to eye dimensions
scalePoints(eye_PupilMinPoints      , eye_ViewBox, eye_Radius)
scalePoints(eye_PupilMaxPoints      , eye_ViewBox, eye_Radius)
scalePoints(eye_IrisPoints          , eye_ViewBox, eye_Radius)
scalePoints(eye_ScleraFrontPoints   , eye_ViewBox, eye_Radius)
scalePoints(eye_ScleraBackPoints    , eye_ViewBox, eye_Radius)
scalePoints(eye_UpperLidClosedPoints, eye_ViewBox, eye_Radius)
scalePoints(eye_UpperLidOpenPoints  , eye_ViewBox, eye_Radius)
scalePoints(eye_UpperLidEdgePoints  , eye_ViewBox, eye_Radius)
scalePoints(eye_LowerLidClosedPoints, eye_ViewBox, eye_Radius)
scalePoints(eye_LowerLidOpenPoints  , eye_ViewBox, eye_Radius)
scalePoints(eye_LowerLidEdgePoints  , eye_ViewBox, eye_Radius)

# Regenerating flexible object geometry (such as eyelids during blinks, or
# iris during pupil dilation) is CPU intensive, can noticably slow things
# down, especially on single-core boards.  To reduce this load somewhat,
# determine a size change threshold below which regeneration will not occur;
# roughly equal to 1/2 pixel, since 2x2 area sampling is used.

# Determine change in pupil size to trigger iris geometry regen
irisRegenThreshold = 0.0
a = pointsBounds(eye_PupilMinPoints) # Bounds of pupil at min size (in pixels)
b = pointsBounds(eye_PupilMaxPoints) # " at max size
maxDist = max(abs(a[0] - b[0]), abs(a[1] - b[1]), # Determine distance of max
              abs(a[2] - b[2]), abs(a[3] - b[3])) # variance around each edge
# maxDist is motion range in pixels as pupil scales between 0.0 and 1.0.
# 1.0 / maxDist is one pixel's worth of scale range.  Need 1/2 that...
if maxDist > 0: irisRegenThreshold = 0.5 / maxDist

# Determine change in eyelid values needed to trigger geometry regen.
# This is done a little differently than the pupils...instead of bounds,
# the distance between the middle points of the open and closed eyelid
# paths is evaluated, then similar 1/2 pixel threshold is determined.
upperLidRegenThreshold = 0.0
lowerLidRegenThreshold = 0.0
p1 = eye_UpperLidOpenPoints[len(eye_UpperLidOpenPoints) / 2]
p2 = eye_UpperLidClosedPoints[len(eye_UpperLidClosedPoints) / 2]
dx = p2[0] - p1[0]
dy = p2[1] - p1[1]
d  = dx * dx + dy * dy
if d > 0: upperLidRegenThreshold = 0.5 / math.sqrt(d)
p1 = eye_LowerLidOpenPoints[len(eye_LowerLidOpenPoints) / 2]
p2 = eye_LowerLidClosedPoints[len(eye_LowerLidClosedPoints) / 2]
dx = p2[0] - p1[0]
dy = p2[1] - p1[1]
d  = dx * dx + dy * dy
if d > 0: lowerLidRegenThreshold = 0.5 / math.sqrt(d)

# Generate initial iris mesh; vertex elements will get replaced on
# a per-frame basis in the main loop, this just sets up textures, etc.
irisMesh = meshInit(32, 4, True, 0, 0.5/eye_IRIS_MAP.iy, False)
irisMesh.set_textures([eye_IRIS_MAP])
irisMesh.set_shader(shader)
irisZ = zangle(eye_IrisPoints, eye_Radius)[0] * 0.99 # Get iris Z depth, for later

irisAnimationFrames = []
eye_ANIMATED_IRIS = conf["eye_ANIMATED_IRIS"]
if eye_ANIMATED_IRIS:
	iFiles = glob.glob(conf["eye_ANIMATED_IRIS_PATTERN"])
	iFiles.sort() # order is vital to animation!
	for f in iFiles:
		irisAnimationFrames.append(pi3d.Texture(f, mipmap=False, filter=pi3d.GL_LINEAR, blend=True))
num_IrisAnimationFrames = len(irisAnimationFrames)


# Eyelid meshes are likewise temporary; texture coordinates are
# assigned here but geometry is dynamically regenerated in main loop.
upperEyelidMesh = meshInit(33, 5, False, 0, 0.5/eye_LID_MAP.iy, True)
upperEyelidMesh.set_textures([eye_LID_MAP])
upperEyelidMesh.set_shader(shader)
lowerEyelidMesh = meshInit(33, 5, False, 0, 0.5/eye_LID_MAP.iy, True)
lowerEyelidMesh.set_textures([eye_LID_MAP])
lowerEyelidMesh.set_shader(shader)

# Generate sclera for eye...start with a 2D shape for lathing...
angle1 = zangle(eye_ScleraFrontPoints, eye_Radius)[1] # Sclera front angle
angle2 = zangle(eye_ScleraBackPoints , eye_Radius)[1] # " back angle
aRange = 180 - angle1 - angle2
pts    = []
for i in range(24):
	ca, sa = pi3d.Utility.from_polar((90 - angle1) - aRange * i / 23)
	pts.append((ca * eye_Radius, sa * eye_Radius))

eyeLathe = pi3d.Lathe(path=pts, sides=64)
eyeLathe.set_textures([eye_SCLERA_MAP])
eyeLathe.set_shader(shader)
reAxis(eyeLathe, 0.0)


# Init global stuff --------------------------------------------------------
thisKeyboard = pi3d.Keyboard() # For capturing key presses

startX       = random.uniform(-30.0, 30.0)
n            = math.sqrt(900.0 - startX * startX)
startY       = random.uniform(-n, n)
destX        = startX
destY        = startY
curX         = startX
curY         = startY
moveDuration = random.uniform(0.075, 0.175)
holdDuration = random.uniform(0.1, 1.1)
startTime    = 0.0
isMoving     = False

frames        = 0
beginningTime = time.time()

eyeLathe.positionX(0.0)
irisMesh.positionX(0.0)
upperEyelidMesh.positionX(0.0)
upperEyelidMesh.positionZ(-eye_Radius - 42)
lowerEyelidMesh.positionX(0.0)
lowerEyelidMesh.positionZ(-eye_Radius - 42)

currentPupilScale  = 0.5
prevPupilScale     = -1.0 # Force regen on first frame
prevUpperLidWeight = 0.5
prevLowerLidWeight = 0.5
prevUpperLidPts    = pointsInterp(eye_UpperLidOpenPoints, eye_UpperLidClosedPoints, 0.5)
prevLowerLidPts    = pointsInterp(eye_LowerLidOpenPoints, eye_LowerLidClosedPoints, 0.5)

ruRegen = True
rlRegen = True

timeOfLastBlink = 0.0
timeToNextBlink = 1.0
blinkState      = 0
blinkDuration   = 0.1
blinkStartTime  = 0

trackingPos = 0.3

thread.start_new_thread(cvThread, ())

nextPupilFactorChange = time.time() + random.uniform(1.0, 4.0)
vFactor = 1.0

nextPupilChange = time.time() - 1

destPupilScale = currentPupilScale

prevBrightness = 0.0

fpsTime = time.time()
framesBefore = 0

sound_MIN_DELAY = conf["sound_MIN_DELAY"]

soundCounter = 0

ambientLoopSound = conf["sound_AMBIENT_LOOP"] 

if (ambientLoopSound != "None"):
	fireloop = pygame.mixer.Sound(ambientLoopSound)
	fireloop.set_volume(conf["sound_AMBIENT_VOLUME"] )
	fireloop.play(-1)

# MAIN LOOP -- runs continuously -------------------------------------------
while True:

	if doPlaySound:
		doPlaySound = False
		timeSinceLastSound = time.time() - lastTimeSound
		if (timeSinceLastSound >= sound_MIN_DELAY):
			soundFile = ""
			while True:
				soundFile = random.choice(glob.glob(sound_FILE_PATTERN))
				if (soundFile != lastSoundFile[0] and soundFile != lastSoundFile[1] and soundFile != lastSoundFile[2] ):
					lastSoundFile[0] = lastSoundFile[1]
					lastSoundFile[1] = lastSoundFile[2]
					lastSoundFile[2] = soundFile
					break
			soundCounter += 1
			print_there(1, 36, "playing sound #%d:   %s                " % ( soundCounter, soundFile ) )
			song = pygame.mixer.Sound(soundFile)
			song.play()
			lastTimeSound = time.time()
		else:
			print_there(1, 35, "cannot play sound #%d, time since last sound is %.2fs                " % ( soundCounter+1, timeSinceLastSound ) )
	
	#add some additional random factor to pupil
	if (time.time() >= nextPupilFactorChange):
		nextPupilFactorChange = time.time() + random.uniform(5.0, 6.0)
		vFactor = random.uniform(0.6, 1.4)
		print_there(1, 12, "vFactor: %.2f                " % ( vFactor ) )
		destPupilScale = destPupilScale * vFactor
		print_there(1, 13, "destPupilScale now: %.2f                " % ( destPupilScale ) )
		if   destPupilScale < eye_PUPIL_MIN: destPupilScale = eye_PUPIL_MIN
		elif destPupilScale > eye_PUPIL_MAX: destPupilScale = eye_PUPIL_MAX

	# brightness changed by more than 10%
	if (abs(prevBrightness - imgBrightness) > 0.1):
		destPupilScale = imgBrightness
		prevBrightness = imgBrightness
		if eye_PUPIL_IN_FLIP: destPupilScale = 1.0 - destPupilScale
		if   destPupilScale < eye_PUPIL_MIN: destPupilScale = eye_PUPIL_MIN
		elif destPupilScale > eye_PUPIL_MAX: destPupilScale = eye_PUPIL_MAX

	pupilDelta = abs(currentPupilScale - destPupilScale)
	if (pupilDelta > 0.01):
		if (time.time() >= nextPupilChange):
			print_there(1, 14, "PUPIL CHANGE     " )
			if currentPupilScale < destPupilScale:
				currentPupilScale += 0.2 * pupilDelta
			else:
				currentPupilScale -= 0.2 * pupilDelta
			nextPupilChange = time.time() + 0.03
	else:
		print_there(1, 14, "no pupil change      " )

	frame(currentPupilScale)

	print_there(1, 15, "currentPupilScale: %.2f         " % ( currentPupilScale ) )
	print_there(1, 16, "destPupilScale: %.2f            " % ( destPupilScale ) )
	print_there(1, 17, "pupilDelta: %.2f                " % ( pupilDelta ) )
	
	pressedKey = thisKeyboard.read()
	if pressedKey == 27:
		break


# clean up and exit
print_there(1, 40, "EXIT  " )

thisKeyboard.close()
DISPLAY.stop()

if (ambientLoopSound != "None"):
	fireloop.stop()

picam.close()

exit(0)