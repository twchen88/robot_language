try:
    import speech_recognition as sr
except: pass      

import time
from threading import Thread

from .evbase import Event
from .events import SpeechEvent
from pywordseg import * # pypi.org/project/pywordseg

#================ Thesaurus ================

class Thesaurus():
    def __init__(self):
        self.words = dict()
        self.add_homophones('卡兹莫', \
                            ["打字幕","叉子没","cosmo", \
                             "卡森莫","毯子没","打字没", "卡提诺"]) # cozmo's chinese name and all the words that google thought I said
        self.add_homophones('右', ['又','誘']) # right, two characters that sound the same
        self.add_homophones('方块一',['分会一'])
        self.add_homophones('方块二',['方块儿']) # cube 2, character that google got confused with every single time (:
        self.add_homophones('方块',['房会', '放会', '分会']) # cube, two phrases with different characters, cube #1
        self.phrase_tree = dict()
        self.add_phrases('方块一',['方块 一','方块蛇']) # cube 1, cube #1
        self.add_phrases('方块二',['方块 二', '方块儿']) # cube 2, cube #2
        self.add_phrases('方块三',['方块 三']) # cube 3, cube #3
        

    def add_homophones(self,word,homophones):
        if not isinstance(homophones,list):
            homophones = [homophones]
        for h in homophones:
            self.words[h] = word

    def lookup_word(self,word):
        return self.words.get(word,word)

    def add_phrases(self,word,phrases):
        if not isinstance(phrases,list):
            phrases = [phrases]
        for phrase in phrases:
            wdict = self.phrase_tree
            for pword in phrase.split(' '):
                wdict[pword] = wdict.get(pword,dict())
                wdict = wdict[pword]
            wdict[''] = word

    def substitute_phrases(self,words):
        result = []
        while words != []:
            word = words[0]
            del words[0]
            wdict = self.phrase_tree.get(word,None) 
            if wdict is None:
                result.append(word)
                continue
            prefix = [word]
            while words != []:
                wdict2 = wdict.get(words[0],None)
                if wdict2 is None: break
                prefix.append(words[0])
                del words[0]
                wdict = wdict2
            subst = wdict.get('',None)
            if subst is not None:
              result.append(subst)
            else:
              result = result + prefix
        print(result)
        return result

#================ SpeechListener ================

class SpeechListener():
    def __init__(self,robot, thesaurus=Thesaurus(), debug=False):
        self.robot = robot
        self.thesaurus = thesaurus
        self.debug = debug
        self.seg = Wordseg(batch_size=64, device="cuda:0", embedding="elmo", elmo_use_cuda=True, mode="TW")

    def speech_listener(self):
        warned_no_mic = False
        print('Launched speech listener.')
        self.rec = sr.Recognizer()
        while True:
            try:
                with sr.Microphone() as source:
                    if warned_no_mic:
                        print('Got a microphone!')
                        warned_no_mic = False
                    while True:
                        if self.debug: print('--> Listening...')
                        try:
                            audio = self.rec.listen(source, timeout=8, phrase_time_limit=8)
                            audio_len = len(audio.frame_data)
                        except:
                            continue
                        if self.debug:
                            print('--> Got audio data: length = {:,d} bytes.'. \
                                  format(audio_len))
                        if audio_len > 1000000: #500000:
                            print('**** Audio segment too long.  Try again.')
                            continue
                        try:
                            utterance = self.rec.recognize_google(audio, language="zh-TW").lower()
                            print("Raw utterance: '%s'" % utterance)
                            words = [self.thesaurus.lookup_word(w) for w in self.seg.cut([utterance])[0]]
                            words = self.thesaurus.substitute_phrases(words)
                            string = "".join(words)
                            print("Heard: '%s'" % string)
                            evt = SpeechEvent(string,words)
                            self.robot.erouter.post(evt)
                        except sr.RequestError as e:
                            print("Could not request results from google speech recognition service; {0}".format(e)) 
                        except sr.UnknownValueError:
                            if self.debug:
                                print('--> Recognizer found no words.')
                        except Exception as e:
                            print('Speech recognition got exception:', repr(e))
            except OSError as e:
                if not warned_no_mic:
                    print("Couldn't get a microphone:",e)
                    warned_no_mic = True
                time.sleep(10)

    def start(self):
        self.thread = Thread(target=self.speech_listener)
        self.thread.daemon = True #ending fg program will kill bg program
        self.thread.start()
