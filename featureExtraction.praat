###########################################################################
#                                                                         #
#  Praat Script Syllable Nuclei                                           #
#  Copyright (C) 2008  Nivja de Jong and Ton Wempe                        #
#                                                                         #
#    This program is free software: you can redistribute it and/or modify #
#    it under the terms of the GNU General Public License as published by #
#    the Free Software Foundation, either version 3 of the License, or    #
#    (at your option) any later version.                                  #
#                                                                         #
#    This program is distributed in the hope that it will be useful,      #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of       #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        #
#    GNU General Public License for more details.                         #
#                                                                         #
#    You should have received a copy of the GNU General Public License    #
#    along with this program.  If not, see http://www.gnu.org/licenses/   #
#                                                                         #
###########################################################################
#
# modified 2010.09.17 by Hugo Quen��, Ingrid Persoon, & Nivja de Jong
# Overview of changes: 
# + change threshold-calculator: rather than using median, use the almost maximum
#     minus 25dB. (25 dB is in line with the standard setting to detect silence
#     in the "To TextGrid (silences)" function.
#     Almost maximum (.99 quantile) is used rather than maximum to avoid using
#     irrelevant non-speech sound-bursts.
# + add silence-information to calculate articulation rate and ASD (average syllable
#     duration.
#     NB: speech rate = number of syllables / total time
#         articulation rate = number of syllables / phonation time
# + remove max number of syllable nuclei
# + refer to objects by unique identifier, not by name
# + keep track of all created intermediate objects, select these explicitly, 
#     then Remove
# + provide summary output in Info window
# + do not save TextGrid-file but leave it in Object-window for inspection
#     (if requested in startup-form)
# + allow Sound to have starting time different from zero
#      for Sound objects created with Extract (preserve times)
# + programming of checking loop for mindip adjusted
#      in the orig version, precedingtime was not modified if the peak was rejected !!
#      var precedingtime and precedingint renamed to currenttime and currentint
#
# + bug fixed concerning summing total pause, feb 28th 2011
###########################################################################

# Augmented by linus.manser@uzh.ch in order to extract additional features:
# meanf0, jit_loc, jit_rap, shim_loc, shim_apq3, hnr, nhr

# counts syllables of all sound utterances in a directory
# NB unstressed syllables are sometimes overlooked
# NB filter sounds that are quite noisy beforehand
# NB use Silence threshold (dB) = -25 (or -20?)
# NB use Minimum dip between peaks (dB) = between 2-4 (you can first try;
#                                                      For clean and filtered: 4)


form Pythia Feature Extraction
   real Silence_threshold_(dB) -25
   real Minimum_dip_between_peaks_(dB) 2
   real Minimum_pause_duration_(s) 0.3
   real Keep_Soundfiles_and_Textgrids 0
   sentence directory /Users/linusmanser/cl/bachelor/pythia2/appdata/segments/
   sentence outDir /Users/linusmanser/cl/bachelor/pythia2/appdata/ext/
   sentence sampleid somethingwentwrong
endform

# shorten variables
silencedb = silence_threshold
mindip = minimum_dip_between_peaks
showtext = keep_Soundfiles_and_Textgrids
minpause = minimum_pause_duration

# read files
	sound = Read from file... 'directory$''sampleid$'.wav
	selectObject: sound

# scale intensity
	sound = Scale intensity: 70.0

# prepare outname
	outName$ = sampleid$

# extract jitter and shimmer
   selectObject: sound
   pid = To Pitch (cc)... 0.0 75.0 15 no 0.03 0.45 0.01 0.35 0.14 600.0
   selectObject: sound
   plusObject: pid
   prid = To PointProcess (cc)
   selectObject: sound
   plusObject: pid
   plusObject: prid
   voiceReport$ = Voice report... 0 0 75 600 1.3 1.6 0.03 0.45
   jit_loc = extractNumber (voiceReport$, "Jitter (local): ")
   jit_loc_abs = extractNumber (voiceReport$, "Jitter (local, absolute): ")
   jit_rap = extractNumber (voiceReport$, "Jitter (rap): ")
   jit_ppq5 = extractNumber (voiceReport$, "Jitter (ppq5): ")
   jit_ddp = extractNumber (voiceReport$, "Jitter (ddp): ")
   shim_loc = extractNumber (voiceReport$, "Shimmer (local): ")
   shim_loc_db = extractNumber (voiceReport$, "Shimmer (local, dB): ")
   shim_apq3 = extractNumber (voiceReport$, "Shimmer (apq3): ")
   shim_apq5 = extractNumber (voiceReport$, "Shimmer (apq5): ")
   shim_apq11 = extractNumber (voiceReport$, "Shimmer (apq11): ")
   shim_dda = extractNumber (voiceReport$, "Shimmer (dda): ")
   pitch_stdev = extractNumber (voiceReport$, "Standard deviation: ")
   pitch_min = extractNumber (voiceReport$, "Minimum pitch: ")
   pitch_max = extractNumber (voiceReport$, "Maximum pitch: ")
   if pitch_max = undefined
		pitch_range = undefined
   else
	    pitch_range = 'pitch_max' - 'pitch_min'
   endif

# extract voiced/unvoiced parts
	windowLength = 2/75
	selectObject: sound
	pitch1id = To Pitch: 0, 75, 500
	pitch2id = Kill octave jumps
	pitchTextGrid = To TextGrid: "voicing", ""
	selectObject: pitch1id
	pitch3id = Smooth: 10
	pitchduration = Get total duration
	selectObject: pitch1id
	pitchTierid = Down to PitchTier
	nPoints = Get number of points
	nVoicedIntervals = 0
	nUnvoicedIntervals = 0
	durationVoiced = 0
	for iPoint from 1 to nPoints
		selectObject: pitchTierid
		startPoint = iPoint
		startTime = Get time from index: startPoint
		voicedPointCounter = 0
		repeat
			time1 = Get time from index: iPoint
			time2 = Get time from index: iPoint+1
			deltaN = time2 - time1
			# printline 'deltaN', 'windowLength', 'iPoint', 'nPoints'
			iPoint+= 1
			voicedPointCounter+=1
			if iPoint >= nPoints
				deltaN = windowLength + 1
			endif
		until deltaN>windowLength
		
		endPoint = startPoint + voicedPointCounter - 1
		endTime = Get time from index: endPoint

		if voicedPointCounter >= 3
			selectObject: pitchTextGrid
			Insert boundary: 1, startTime
			Insert boundary: 1, endTime
			interval = Get interval at time: 1, startTime+(endTime-startTime)/2
			Set interval text: 1, interval, "v"
			nVoicedIntervals += 1
			intervalDuration = endTime - startTime
			durationVoiced += intervalDuration
		endif

		iPoint-=1
		
	endfor
	selectObject: pitchTextGrid
	nIntervals = Get number of intervals: 1
	for iInterval to nIntervals
		label$ = Get label of interval: 1, iInterval
		if label$ = ""
			Set interval text: 1, iInterval, "u"
			nUnvoicedIntervals += 1
		endif
	endfor

	vCounter = 0
	uCounter = 0
	# select try
	select pitchTextGrid
	# printline 'pitchTextGrid' test test
	pVO = 'durationVoiced' / 'pitchduration'
	deltaUV = undefined
	deltaVO = undefined
	meanUV = undefined
	meanVO = undefined
	varcoUV = undefined
	varcoVO = undefined
	if nVoicedIntervals > 0
		if nUnvoicedIntervals > 0
			voicedintervals# = zero#(nVoicedIntervals)
			unvoicedintervals# = zero#(nUnvoicedIntervals)
			for iInterval from 1 to nIntervals
				intervallabel$ = Get label of interval: 1, iInterval
				intervalStartTime = Get start time of interval: 1, iInterval
				intervalEndTime = Get end time of interval: 1, iInterval
				intervalDuration = intervalEndTime - intervalStartTime
				# VOICED intervals
				if intervallabel$ = "v"
					vCounter += 1
					voicedintervals#[vCounter] = intervalDuration
				elif intervallabel$ = "u"
					uCounter += 1
					unvoicedintervals#[uCounter] = intervalDuration
				endif
				deltaUV = stdev(unvoicedintervals#)
				deltaVO = stdev(voicedintervals#)
				meanUV = mean(unvoicedintervals#)
				meanVO = mean(voicedintervals#)
				varcoUV = 100*deltaUV/meanUV
				varcoVO = 100*deltaVO/meanVO
			endfor
		else
			
		endif
	endif

# extract pitch measurements
	# pitch_mean
	selectObject: sound
	soundpitch = To Pitch: 0.0, 75, 600
	pitchtier = Down to PitchTier
	pitchtor = Down to TableOfReal: "Hertz"
	pitchtable = To Table: ""
	Set column label (index): 1, "todelete"
	Remove column: "todelete"
	Remove column: "Time"
	pitch_mean = Get mean: "F0"
	Save as comma-separated file: outDir$ + outName$ + "_pitchtable.csv"


# extract spectral tilt etc.
	rowcount = 0
	spectableid = Create Table with column names: "spectable", rowcount, "stilt skurt scog bandenergylow bandenergyhigh vlhr"
	selectObject: sound
	totaldur = Get total duration
	starttime = 0
	framesize = 0.01
	while starttime + framesize < totaldur
		selectObject: sound
		snippet = Extract part: starttime, starttime + framesize, "rectangular", 1.0, "yes"
		snippet_ltasid = To Ltas: 100
		bandenergylow = Get mean: 2000, 4500, "energy"
		bandenergyhigh = Get mean: 4500, 8000, "energy"
		stiltreport$ = Report spectral tilt: 100, 5500, "Linear", "Robust"
		stilt = extractNumber (	stiltreport$, "Slope: ")
		selectObject: snippet
		snippet_spectrumid = To Spectrum: "yes"
		skurt = Get kurtosis: 2.0
		scog = Get centre of gravity: 2.0
		if pitch_mean != undefined
			fcut = 'pitch_mean' * 4.47
			ltenergy = Get band energy: 65, 'fcut' 
			htenergy = Get band energy: 'fcut' , 8000
			vlhr = 'ltenergy' / 'htenergy'
		else
			fcut = undefined
			ltenergy = undefined
			htenergy = undefined
			vlhr = undefined
		endif

		selectObject: spectableid
		Append row
		rowcount += 1
		Set numeric value: rowcount, "stilt", stilt
		Set numeric value: rowcount, "skurt", skurt
		Set numeric value: rowcount, "scog", scog
		Set numeric value: rowcount, "bandenergylow", bandenergylow
		Set numeric value: rowcount, "bandenergyhigh", bandenergyhigh
		Set numeric value: rowcount, "vlhr", vlhr
		starttime += framesize
		# midway clean up
		select 'snippet'
		plus 'snippet_ltasid'
		plus 'snippet_spectrumid'
		Remove
	endwhile
	selectObject: spectableid
	Save as comma-separated file: outDir$ + outName$ + "_spectable.csv"
	
# extract Harmonicity (HNR)
	selectObject: sound
	soundharmonicity = To Harmonicity (cc): 0.01, 75.0, 0.1, 1.0
	selectObject: soundharmonicity
	harmatrix = To Matrix
	selectObject: harmatrix
	harmatrix_trans = Transpose
	selectObject: harmatrix_trans
	harmtor = To TableOfReal
	harmtable = To Table: ""
	Set column label (index): 1, "todelete"
	Set column label (index): 2, "harmonicity"
	Remove column: "todelete"
	Save as comma-separated file: outDir$ + outName$ + "_harmtable.csv"

# extract formant frequencies
	selectObject: sound
	soundformants = To Formant (burg): 0.0, 5.0, 5500.0, 0.025, 50.0
	soundtable = Down to Table: "no", "yes", 6, "no", 3, "yes", 6, "yes"
	
# extract formant amplitudes
	selectObject: sound
	soundspectrogram = To Spectrogram: 0.005, 5000.0, 0.002, 20.0, "Gaussian"
	# use the time and frequencies of the formant frequncies to extract the
	# corresponding amplitude value
	selectObject: soundtable
	soundtable = Extract rows where column (text): "F1(Hz)", "is not equal to", "--undefined--"
	soundtable = Extract rows where column (text): "F2(Hz)", "is not equal to", "--undefined--"
	soundtable = Extract rows where column (text): "F3(Hz)", "is not equal to", "--undefined--"
	soundtable = Extract rows where column (text): "F4(Hz)", "is not equal to", "--undefined--"
	Append column: "f1amp"
	Append column: "f2amp"
	Append column: "f3amp"
	Append column: "f4amp"
	numberofrows = Get number of rows
	for row from 1 to numberofrows
		selectObject: soundtable
		time = Get value: row, "time(s)"
		ff1 = Get value: row, "F1(Hz)"
		ff2 = Get value: row, "F2(Hz)"
		ff3 = Get value: row, "F3(Hz)"
		ff4 = Get value: row, "F4(Hz)"
		selectObject: soundspectrogram
		f1amp = Get power at: time, ff1
		f2amp = Get power at: time, ff2
		f3amp = Get power at: time, ff3
		f4amp = Get power at: time, ff4
		selectObject: soundtable
		Set numeric value: row, "f1amp", f1amp
		Set numeric value: row, "f2amp", f2amp
		Set numeric value: row, "f3amp", f3amp
		Set numeric value: row, "f4amp", f4amp
	endfor
	selectObject: soundtable
	Append difference column: "f1amp" , "f2amp", "I12diff"
	Append difference column: "f2amp" , "f3amp", "I23diff"
	Remove column: "B1(Hz)"
	Remove column: "B2(Hz)"
	Remove column: "B3(Hz)"
	Remove column: "B4(Hz)"
	Remove column: "B5(Hz)"
	Remove column: "F5(Hz)"
	Remove column: "nformants"
	Remove column: "time(s)"
	Save as comma-separated file: outDir$ + outName$ + "_formanttable.csv"

# for speakingrate, speakingratio (taken from original script)
# use object ID
   selectObject: sound
   soundname$ = selected$("Sound")
   # soundid = selected("Sound")
   selectObject: sound

   originaldur = Get total duration
   # allow non-zero starting time
   bt = Get starting time

   # Use intensity to get threshold
   To Intensity... 50 0 yes
   intid = selected("Intensity")
   start = Get time from frame number... 1
   nframes = Get number of frames
   end = Get time from frame number... 'nframes'

   # estimate noise floor
   minint = Get minimum... 0 0 Parabolic
   # estimate noise max
   maxint = Get maximum... 0 0 Parabolic
   #get .99 quantile to get maximum (without influence of non-speech sound bursts)
   max99int = Get quantile... 0 0 0.99

   # estimate Intensity threshold
   threshold = max99int + silencedb
   threshold2 = maxint - max99int
   threshold3 = silencedb - threshold2
   if threshold < minint
       threshold = minint
   endif

  # get pauses (silences) and speakingtime
   To TextGrid (silences)... threshold3 minpause 0.1 silent sounding
   textgridid = selected("TextGrid")
   silencetierid = Extract one tier... 1
   selectObject: silencetierid
   # printline the selected id is 'silencetierid'
   nSounding = Count intervals where: 1, "is equal to", "sounding"
   # printline 'nSounding'
   if 'nSounding' >= 1
       select silencetierid
	   soundingtableid = Down to Table: "no", 6, "yes", "no"
       # printline --> 'soundingtableid'
       selectObject: soundingtableid
       soundingtimetableid = Extract rows where column (text): "text", "is equal to", "sounding"
	   # compute sounding intervals
	   selectObject: soundingtimetableid
	   nSounding = Get number of rows
	   speakingtot = 0
   	   for iSound from 1 to nSounding
			starttime = Get value: iSound, "tmin"
			endtime = Get value: iSound, "tmax"
			speakingdur = 'endtime' - 'starttime'
      		speakingtot = 'speakingdur' + 'speakingtot'
   	   endfor
   endif

   select 'intid'
   Down to Matrix
   matid = selected("Matrix")
   # Convert intensity to sound
   To Sound (slice)... 1
   sndintid = selected("Sound")

   # use total duration, not end time, to find out duration of intdur
   # in order to allow nonzero starting times.
   intdur = Get total duration
   intmax = Get maximum... 0 0 Parabolic

   # estimate peak positions (all peaks)
   To PointProcess (extrema)... Left yes no Sinc70
   ppid = selected("PointProcess")

   numpeaks = Get number of points

   # fill array with time points
   for i from 1 to numpeaks
       t'i' = Get time from index... 'i'
   endfor 

   # fill array with intensity values
   select 'sndintid'
   peakcount = 0
   for i from 1 to numpeaks
       value = Get value at time... t'i' Cubic
       if value > threshold
             peakcount += 1
             int'peakcount' = value
             timepeaks'peakcount' = t'i'
       endif
   endfor

   # fill array with valid peaks: only intensity values if preceding 
   # dip in intensity is greater than mindip
   select 'intid'
   validpeakcount = 0
   currenttime = timepeaks1
   currentint = int1
   for p to peakcount-1
      following = p + 1
      followingtime = timepeaks'following'
      dip = Get minimum... 'currenttime' 'followingtime' None
      diffint = abs(currentint - dip)
      if diffint > mindip
         validpeakcount += 1
         validtime'validpeakcount' = timepeaks'p'
      endif
         currenttime = timepeaks'following'
         currentint = Get value at time... timepeaks'following' Cubic
   endfor
   # Look for only voiced parts
   selectObject: sound 
   To Pitch (ac)... 0.02 30 4 no 0.03 0.25 0.01 0.35 0.25 450
   # keep track of id of Pitch
   pitchid = selected("Pitch")
   voicedcount = 0
   for i from 1 to validpeakcount
      querytime = validtime'i'
      select 'textgridid'
      whichinterval = Get interval at time... 1 'querytime'
      whichlabel$ = Get label of interval... 1 'whichinterval'
      select 'pitchid'
      value = Get value at time... 'querytime' Hertz Linear
      if value <> undefined
         if whichlabel$ = "sounding"
             voicedcount = voicedcount + 1
             voicedpeak'voicedcount' = validtime'i'
         endif
      endif
   endfor
   select 'pitchid'
   pitch_med = Get quantile: 0.0, 0.0, 0.50, "Hertz"  
   # calculate time correction due to shift in time for Sound object versus
   # intensity object
   timecorrection = originaldur/intdur
   # Insert voiced peaks in TextGrid
   if showtext > 0
      select 'textgridid'
      Insert point tier... 1 syllables
      for i from 1 to voicedcount
          position = voicedpeak'i' * timecorrection
          Insert point... 1 position 'i'
      endfor
   endif



# summarize results in Info window
	speakingrate = 'voicedcount'/'originaldur'
	articulationrate = 'voicedcount'/'speakingtot'
	asd = 'speakingtot'/'voicedcount'
	speakingratio = 'speakingtot'/'originaldur'
	outTable = Create Table with column names: "extraction_data", 1, "soundname pitch_stdev pitch_min pitch_max pitch_range pitch_med jit_loc jit_loc_abs jit_rap jit_ppq5 jit_ddp shim_loc shim_apq3 shim_apq5 shim_dda deltaUV deltaVO meanUV meanVO varcoUV varcoVO speakingrate speakingratio pVO"
	# insert table values
	selectObject: outTable
	Set string value: 1, "soundname", soundname$
	Set numeric value: 1, "pitch_stdev", 'pitch_stdev:4'
	Set numeric value: 1, "pitch_min", 'pitch_min:4'
	Set numeric value: 1, "pitch_max", 'pitch_max:4'
	Set numeric value: 1, "pitch_range", 'pitch_range:4'
	Set numeric value: 1, "pitch_med", 'pitch_med:4'
	Set numeric value: 1, "jit_loc", 'jit_loc:4'
	Set numeric value: 1, "jit_loc_abs", 'jit_loc_abs:4'
	Set numeric value: 1, "jit_rap", 'jit_rap:4'
	Set numeric value: 1, "jit_ppq5", 'jit_ppq5:4'
	Set numeric value: 1, "jit_ddp", 'jit_ddp:4'
	Set numeric value: 1, "shim_loc", 'shim_loc:4'
	Set numeric value: 1, "shim_apq3", 'shim_apq3:4'
	Set numeric value: 1, "shim_apq5", 'shim_apq5:4'
	Set numeric value: 1, "shim_dda", 'shim_dda:4'
	Set numeric value: 1, "deltaUV", 'deltaUV:4'
	Set numeric value: 1, "deltaVO", 'deltaVO:4'
	Set numeric value: 1, "meanUV", 'meanUV:4'
	Set numeric value: 1, "meanVO", 'meanVO:4'
	Set numeric value: 1, "varcoUV", 'varcoUV:4'
	Set numeric value: 1, "varcoVO", 'varcoVO:4'
	Set numeric value: 1, "speakingrate", 'speakingrate:4'
	Set numeric value: 1, "speakingratio", 'speakingratio:4'
	Set numeric value: 1, "pVO", 'pVO:4'
	Save as comma-separated file: outDir$ + outName$ + "_rest.csv"

# Clean-up
	select 'sound'
	plus 'ppid'
	plus 'intid'
	plus 'prid'
	plus 'pid'
	plus 'soundpitch'
	plus 'pitchtier'
	plus 'pitchid'
	plus 'pitchtor'
	plus 'pitchtable'
	plus 'soundharmonicity'
	plus 'harmatrix'
	plus 'harmatrix_trans'
	plus 'harmtor'
	plus 'harmtable'
	plus 'soundformants'
	plus 'soundtable'
	plus 'soundspectrogram'
	plus 'outTable'
	plus 'pitchid'
	plus 'pitch1id'
	plus 'pitch2id'
	plus 'pitch3id'
	plus 'pitchTextGrid'
	plus 'pitchTierid'
	plus 'spectableid'
	plus 'matid'
	plus 'soundpitch'
	plus 'pitchtier'
	plus 'silencetierid'
	plus 'textgridid'
	plus 'sndintid'
	plus 'soundingtimetableid'
	plus 'soundingtableid'
	Remove