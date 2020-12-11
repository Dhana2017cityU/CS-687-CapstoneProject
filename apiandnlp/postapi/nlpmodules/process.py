#import relevant libraries
import spacy
from pathlib import Path
from spacy.matcher import Matcher, PhraseMatcher
import pandas as pd
from PyDictionary import PyDictionary

#-----Load Path-------------------------------------------#

def loadPath():
    #Get parent path
    path = Path(__file__).parent.parent.parent
    #Get spaCy model for English
    enpath = str(path) + ("/en_core_web_sm/en_core_web_sm-2.3.1")
    #Get transitive verb dataset 
    transPath = str(path) + ("/verbdataset.csv")  
    return enpath, transPath

enpath, verbPath = loadPath()
nlp = spacy.load(enpath)
df = pd.read_csv(verbPath)
dictionary=PyDictionary()

#-----Global constants------------------------------------#

commaSeperator =','

#-----Helper methods--------------------------------------#

#validate is value is exists
def is_exists(val, arr):
    breturn = False
    for a in arr:
        if str(a) == str(val):
            breturn = True
            break
    return breturn

#To get root verb
def get_root_verb(nlp_doc):
    
    pattern = [{'DEP': 'ROOT', 'POS': 'VERB'}]
    matcher = Matcher(nlp.vocab)
    matcher.add("Complex", None, pattern)
    matches = matcher(nlp_doc)
    lemmaVal =''
    textVal = ''
    if matches:    
        for match_id, start, end in matches:
            textVal = nlp_doc[start:end].text            
            lemmaVal = nlp_doc[start:end].lemma_              
            break   
    return textVal, lemmaVal
    
#-----Find all ambiguites in a sentence--------------------#

# The method returns direct verb and return string format
def is_indirect_verb(nlp_doc):
    
    textVal, rootVerb = get_root_verb(nlp_doc)
    if len(rootVerb) > 0:
        wordstr = ''     
        if rootVerb not in df.values:
            dictionaryWord = dictionary.synonym(rootVerb)
            
            for word in dictionaryWord:           
                if word in df.values:
                    wordstr = wordstr + word + commaSeperator        
        if len(wordstr) > 0:
            return textVal, wordstr[0:len(wordstr)-1]
        else:
            return textVal, 'nomatch'
    else:
        return textVal, 'nomatch'

#Identify lexical_InsideBehaviour
def lexical_InsideBehaviour(nlp_doc):
    
    strArr = []
    doc = nlp(str(nlp_doc))
    terms = ["unitl", "during", "through", "after", "at"]

    insideBehaviour = [nlp(text) for text in terms]
    
    phrase_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    phrase_matcher.add("insideBehaviour", None, *insideBehaviour)
    matches = phrase_matcher(doc)    
    for match_id, start, end in matches:
        span =  doc[start:end] 
        strArr.append(span.text)    
    return strArr


#Identify Lexcial dangerous plural
def lexical_dangerous(nlp_doc):
    strArr = []
    terms = ["all", "each", "every", "any", "few", "little", "many", "much", "several", "some", "a lot"]
    lexical_dangerous_plural = [nlp(text) for text in terms]
    doc = nlp(str(nlp_doc))
    phrase_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    phrase_matcher.add("lexicalplural", None, *lexical_dangerous_plural)    
    matches = phrase_matcher(doc)

    for match_id, start, end in matches:
        span =  doc[start:end]              
        strArr.append(span.text)
    return strArr

#Identify lexical_weak_phrase
def lexical_weak_phrase(nlp_doc):
    strArr = []
    terms = ["can", "could", "may", "might", "ought to", "preferred", "would", "adequate", "as appropriate", "be capable of", "capability of", "capability to", "effective","as required","normal","provide for","timely","easy to", "Optionally"]    
    lexical_weak_phrase = [nlp(text) for text in terms]
    doc = nlp(str(nlp_doc))
    phrase_matcher = PhraseMatcher(nlp.vocab, attr='LOWER') 
    phrase_matcher.add("weakphrase", None, *lexical_weak_phrase)
    matches = phrase_matcher(doc)

    for match_id, start, end in matches:
        span =  doc[start:end]             
        strArr.append(span.text)
    return strArr

#Identify lexical_unnecessary
def lexical_unnecessary(nlp_doc):
    strArr = []
    terms = ["usually", "normally"]
    unnecessary = [nlp(text) for text in terms]
    doc = nlp(str(nlp_doc))
    phrase_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    phrase_matcher.add("unnecessary", None, *unnecessary)
    matches = phrase_matcher(doc)

    for match_id, start, end in matches:      
        span =  doc[start:end]             
        strArr.append(span.text)
    
    return strArr


#-----Find all incompletes in a sentence-------------------#

#Identify Incompletes
def sentence_Incompletes(nlp_doc):
    strArr = []
    terms = ["TBD", "TBS", "TBE", "TBC", "TBR", "not defined", "not determined", "but not limited to", "as a minimum"]
    incompletes = [nlp(text) for text in terms]
    doc = nlp(str(nlp_doc))
    phrase_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    phrase_matcher.add("Incompletes", None, *incompletes)
    matches = phrase_matcher(doc)

    for match_id, start, end in matches:      
        span =  doc[start:end]             
        strArr.append(span.text)    
    return strArr 

#-----Find all imperative in a sentence--------------------#
def is_imperative_word(nlp_doc):
    strArr = []
    terms = ["must", "is required to", "are applicable", "are to", "responsible for"]
    imperative = [nlp(text) for text in terms]
    doc = nlp(str(nlp_doc))
    phrase_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    phrase_matcher.add("imperative", None, *imperative)
    matches = phrase_matcher(doc)

    for match_id, start, end in matches:      
        span =  doc[start:end]             
        strArr.append(span.text)    
    return strArr 

#-----Find all Continuance--------------------------------#  

def is_continuance_word(nlp_doc):
    strArr = []
    terms = ["below:","as follows:","following:", "listed:", "in particular:", "support:", " and ", ":"]
    continuance = [nlp(text) for text in terms]
    doc = nlp(str(nlp_doc))
    phrase_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    phrase_matcher.add("continuance", None, *continuance)
    matches = phrase_matcher(doc)

    for match_id, start, end in matches:      
        span =  doc[start:end]             
        strArr.append(span.text)    
    return strArr 
    

#-----Find direct object-----------------------------------#  

def is_directive_word(nlp_doc):
    strArr = []
    terms = ["e.g.", "i.e.", "For example", "Figure", "Note:"]
    directive = [nlp(text) for text in terms]
    doc = nlp(str(nlp_doc))
    phrase_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    phrase_matcher.add("directive", None, *directive)
    matches = phrase_matcher(doc)

    for match_id, start, end in matches:      
        span =  doc[start:end]             
        strArr.append(span.text)    
    return strArr 
    
#-----Find forward '/'-------------------------------------#

def is_forwardslash_find(nlp_doc):
    result = str(nlp_doc).find('/')
    if result == -1:
       return False, 'nomatch'
    else:
        return True, str(nlp_doc)[result]

#-----Process return string - apply Bot Markup syntax------#

def return_processstring(strString, arrayOfLex):
    arr = strString.split(' ')
    arr1 = []
    for a in arr:     
        if str(a) in arrayOfLex:   
            a = f'<strong>{a}</strong>' 
        arr1.append(a)
    rejoin = ' '.join(arr1)
    return rejoin
 
#------process ambiguity-----------------------------------#

def get_output_sentence(nlp_doc):
    
    strReturn=str(nlp_doc)
    strMessageCollection = []
    existingList = []
    
    indirectVerb, stringValue = is_indirect_verb(nlp_doc)       
    if stringValue != 'nomatch':
        replaceString1 = f'**{indirectVerb}**'
        strReturn = strReturn.replace(indirectVerb, replaceString1)
        strMessageCollection.append(f'<ul><li>Reconsider using the following action verbs instead of <strong>{indirectVerb}</strong>: {stringValue}</li></ul>\n')

    isForwardSlash, stringValue = is_forwardslash_find(nlp_doc)
    if isForwardSlash:        
        replaceString2 = f'<strong>{stringValue}</strong>'        
        strReturn = strReturn.replace(stringValue, replaceString2)       
        strMessageCollection.append(f'<ul><li>Avoid writing X{replaceString2}Y. Instead write it as X <strong>or</strong> Y</li></ul>\n')

    arrayOfLex = lexical_dangerous(nlp_doc)
    strLex = ''
    if len(arrayOfLex) > 0:
        strReturn = return_processstring(strReturn, arrayOfLex)
        for item in arrayOfLex:         
           if item.lower() not in existingList:
                existingList.append(item.lower())
                strLex = strLex + ' ' + item.lower()
        strMessageCollection.append(f'<ul><li>Avoid using the word <strong>{strLex}</strong>. It is an ambiguous word classified as <strong>potentially dangerous plural.</strong></li></ul>\n')
    
    arrayOfLex = lexical_weak_phrase(nlp_doc)
    strLex = ''
    if len(arrayOfLex) > 0:
        strReturn = return_processstring(strReturn, arrayOfLex)
        for item in arrayOfLex:           
           if item.lower() not in existingList:
                existingList.append(item.lower())
                strLex = strLex + ' ' + item.lower()
        strMessageCollection.append(f'<ul><li>Avoid using the word <strong>{strLex}</strong>. It is an ambiguous word classified as <strong>weak phrase</strong>.</li></ul>\n')

    arrayOfLex = lexical_unnecessary(nlp_doc)
    strLex = ''
    if len(arrayOfLex) > 0:
        strReturn = return_processstring(strReturn, arrayOfLex)
        for item in arrayOfLex:          
           if item.lower() not in existingList:
                existingList.append(item.lower())
                strLex = strLex + ' ' + item.lower()
        strMessageCollection.append(f'<ul><li>Avoid using the word <strong>{strLex}</strong>. It is an ambiguous word.</li></ul>\n')
    
    #Incomplete
    arrayOfLex = sentence_Incompletes(nlp_doc)
    strLex = ''
    if len(arrayOfLex) > 0:        
        strReturn = return_processstring(strReturn, arrayOfLex)        
        for item in arrayOfLex:           
           if item.lower() not in existingList:
                existingList.append(item.lower())
                strLex = strLex + ' ' + item
        strMessageCollection.append(f'<ul><li>Use of <strong>{strLex}</strong> indicates the requirement is incomplete.</li></ul>\n')

    #imperative
    arrayOfLex = is_imperative_word(nlp_doc)
    strLex = ''
    if len(arrayOfLex) > 0:
        strReturn = return_processstring(strReturn, arrayOfLex)
        for item in arrayOfLex:           
           if item.lower() not in existingList:
                existingList.append(item.lower())
                strLex = strLex + ' ' + item.lower()
        strMessageCollection.append(f'<ul><li>The requirement semantic structure does not allow <strong>{strLex}</strong> as an imperative. The allowed imperatives are <strong>shall</strong> (legally binding), <strong>should</strong> (strongly recommended), and <strong>will</strong>.</li></ul>\n')


    return strReturn, strMessageCollection

#-----Find complex sentence ------------------------------#

def is_complex_sentence(nlp_doc):

    strReturn = str(nlp_doc)
    pattern = [{'POS': 'VERB'}]
    matcher = Matcher(nlp.vocab)
    matcher.add("Complex", None, pattern)
    matches = matcher(nlp_doc)
    arr = []

    for match_id, start, end in matches:        
        arr.append(nlp_doc[start:end])
   
    if len(arr) > 4:
        return f'{strReturn}  <ul><li>The sentence has multiple verbs.Reconsider split the sentence into <strong>small</strong> sentences.</li></ul>' , True
    else:
        return '', False

#------Find correct sentence structure used---------------#

def is_good_structure_followed(nlp_doc):

    strGood=str(nlp_doc)
    strMessageCollection = []
    existingList = []

    arrayOfLex = is_continuance_word(nlp_doc)
    strLex = ''
    if len(arrayOfLex) > 0:
        strGood = return_processstring(strGood, arrayOfLex)
        for item in arrayOfLex:           
           if item.lower() not in existingList:
                existingList.append(item.lower())
                strLex = strLex + ' ' + item.lower()
        strMessageCollection.append(f'<ul><li>Continuance phrase <strong>{strLex}</strong>. The extent that CONTINUANCES are used is an indication that requirements have been organized and structured.</li></ul>\n')

    arrayOfLex = is_directive_word(nlp_doc)
    strLex = ''
    if len(arrayOfLex) > 0:
        strGood = return_processstring(strGood, arrayOfLex)
        for item in arrayOfLex:           
           if item.lower() not in existingList:
                existingList.append(item.lower())
                strLex = strLex + ' ' + item.lower()
        strMessageCollection.append(f'<ul><li>Good use of directives <strong>{strLex}</strong>. It helps to define precise requirement.</li></ul>\n')

    strMessageCollection.append(f'<ul><li>Success: It is a correct statement.</li></ul>\n')
    return strGood, strMessageCollection

#-----create warning messages-----------------------------#
def is_acronym_find(nlp_doc):
    pattern = [{'POS': 'PROPN', 'TAG': 'NNP'}]

    matcher = Matcher(nlp.vocab)
    matcher.add("acronym", None, pattern)
    matches = matcher(nlp_doc)
    arr = []
    for match_id, start, end in matches:
        span = nlp_doc[start:end]
        if str(span).isupper():        
            arr.append(span.text)   
    return arr

#-----Find incomplete--------------------------------------#
def is_max_min_find(nlp_doc):
    strArr = []
    terms = ["maximum", "minimum", "max.", "min."]
    maxAndmin = [nlp(text) for text in terms]
    doc = nlp(str(nlp_doc))
    phrase_matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    phrase_matcher.add("maxAndmin", None, *maxAndmin)
    matches = phrase_matcher(doc)

    for match_id, start, end in matches:      
        span =  doc[start:end]             
        strArr.append(span.text)    
    return strArr 
    


def is_warning_find(nlp_doc):
    
    strReturn=str(nlp_doc)
    strMessageCollection = []
    existingList = []

    arrayOfWarning = lexical_InsideBehaviour(nlp_doc)
    strLex = ''
    if len(arrayOfWarning) > 0:
        strReturn = return_processstring(strReturn, arrayOfWarning)
        for item in arrayOfWarning:           
           if item.lower() not in existingList:
                existingList.append(item.lower())
                strLex = strLex + ' ' + item.lower()
        strMessageCollection.append(f'<ul><li>Warning: Use of <strong>{strLex}</strong> could potentially bring ambiguity to the statement.</li></ul>\n')

    arrayOfWarning = is_acronym_find(nlp_doc)
    strLex = ''
    if len(arrayOfWarning) > 0:
        strReturn = return_processstring(strReturn, arrayOfWarning)
        for item in arrayOfWarning:           
           if item.lower() not in existingList:
                existingList.append(item.lower())                 
                strLex = strLex + ' ' + item
        strMessageCollection.append(f'<ul><li>Warning: Make sure acronyms are expanded. <strong>{strLex}</strong></li></ul>\n')
    
    arrayOfWarning = is_max_min_find(nlp_doc)
    strLex = ''
    if len(arrayOfWarning) > 0:
        strReturn = return_processstring(strReturn, arrayOfWarning)
        for item in arrayOfWarning:           
           if item.lower() not in existingList:
                existingList.append(item.lower())
                strLex = strLex + ' ' + item
        strMessageCollection.append(f'<ul><li>Warning: Avoid using terms "maximum" or "minimum". The terms maximum or minimum leaves an open interpretation which leads to subjective judgment on the allowed unit quantity, and consequently contributes to vagueness. Thus, write down the exact measurable or at least tolerable unit quantity to avoid vagueness introduced by maximum or minimum.</li></ul>\n')

    return strReturn, strMessageCollection

#-----Find Boilerplates and INCOSE Rule--------------------#
def INCOSE_rule(nlp_doc):
   
    pattern1 = [{'DEP': 'aux'}, {'DEP': 'neg'}, {'TAG': 'VB'}]
    pattern2 = [{'DEP': 'aux'},{'DEP': 'ROOT'},{'TAG': 'JJ'}, {'TAG': 'TO'}, {'TAG': 'VB'}]
    pattern3 = [{'DEP': 'aux'}, {'POS': 'VERB'},{'TAG': 'NN'},{'TAG': 'IN'},{'TAG': 'DT'},{'TAG': 'NN'}, {'TAG': 'TO'}]
    pattern4 = [{'DEP': 'aux'}, {'POS': 'VERB'}]
    pattern5 = [{'DEP': 'aux'}, {'TAG': 'VB'}, {'DEP': 'ROOT'},]

    matcher = Matcher(nlp.vocab)
    matcher.add("RuleOne", None, pattern1, pattern2, pattern3, pattern4, pattern5)
    matches = matcher(nlp_doc) 
    isVerb = False
    isSubject = False
    isObject = False

    verbPos = 0
    returnReason =''

    #find verb clause  
    for match_id, start, end in matches:
        verbPos = start
        span = nlp_doc[start:end]        
        if span:
            isVerb = True
            break                  

    #find noun clause
    noun = ''
    for token in nlp_doc:       
        if token.dep_ == 'nsubjpass' or token.dep_ == 'nsubj':
           if token.i < verbPos:            
                noun = token
                break
    if noun:
        isSubject = True

    #find direct object clause
    for token in nlp_doc:        
        if token.dep_ == 'dobj' or token.dep_ == 'pobj':
            if token.i > verbPos:
                isObject = True            
                break 
    
    if isVerb == False:
        returnReason = 'Error: The sentence does not comply to the requirement template.'        
   
    if isSubject == False:
        returnReason = 'Error: The sentence does not comply to the requirement template.'

    if isObject == False:
        returnReason = 'Error: The sentence does not comply to the requirement template.'

    if isVerb == True and isSubject == True and isObject == True:
        return True, verbPos,returnReason

    return False, verbPos, returnReason


# ----- Start Rule_one Event-driven requirements | Behavior driven requirements | State-driven requirements---#

def is_startSentence(nlp_doc):
    pattern = [{'TEXT': {'REGEX': '(?i)^(when|if|while)*$'}}]
    matcher = Matcher(nlp.vocab)
    matcher.add('startword', None, pattern)
    matches = matcher(nlp_doc)    
    if matches:
        return True
    else:
        return False

def is_valid_drivenRequirements(nlp_doc):   
    pattern = [{'TEXT': {'REGEX': '(?i)^(when|if|while)*$'}}]
    pattern1 = ''
    matcher = Matcher(nlp.vocab)
    matcher.add('validdriven', None, pattern)
    matches = matcher(nlp_doc)
    first_word = str(nlp_doc[0]).lower()       
    if matches:
        if first_word == 'when':
            pattern1 = [{'TAG': 'NNP'},{'TAG': ','}]
        if first_word == 'if':
            pattern1 = [{'TAG': 'VBN'},{'TAG': ','}]
        if first_word == 'while':
            pattern1 = [{'TAG': 'NN'},{'TAG': ','}]
        matcher.remove('validdriven')
        matcher.add("commapostion", None, pattern1)
        matches = matcher(nlp_doc)        
        if matches:
           return True
        else:           
            return False        
    else:
        return False

#-----Main method for Processing Sentence------------------#

#Main method to verify ambiguity
def processText(text):

    nlp_doc = nlp(text)
    returnArr = []   
    finalreturn = ''
    processreturn = ''
    arryofReturnString = []

    isEndOfError = False
    reasonText = ''
    # empty dictionary
    my_dict = {}
         
    for sent in nlp_doc.sents:
            
        returnValue, breturn = is_complex_sentence(sent)
        if breturn:
            print(returnValue)
            return returnValue
        else:
            incomingReturn, returnArr = get_output_sentence(sent)
            if len(returnArr) > 0:       
                my_dict[incomingReturn] = returnArr
            else:
                isRuleOne, verbPos, reasonText = INCOSE_rule(sent)
                incomingWarning, warningArr = is_warning_find(nlp_doc)
                               
                if isRuleOne:
                    #incomingWarning, warningArr = is_warning_find(nlp_doc)
                    incomingGoodReturn, goodReturnArr = is_good_structure_followed(sent)

                    if is_startSentence(sent):
                        isValid = is_valid_drivenRequirements(sent[0:verbPos])                        
                        if isValid:
                            if len(warningArr) > 0:
                                my_dict[incomingWarning] = warningArr
                            else:  
                                if len(goodReturnArr) > 0:
                                    my_dict[incomingGoodReturn] = goodReturnArr 
                        else:                            
                            arryofReturnString.append(f'{sent} <ul><li>Reconsider placing comma on the sentence since it starts with <strong>{sent[0]}</strong>.</li></ul>')
                    else:
                        if len(warningArr) > 0:
                            my_dict[incomingWarning] = warningArr
                        else:  
                            if len(goodReturnArr) > 0:
                                my_dict[incomingGoodReturn] = goodReturnArr 
                else:
                    my_dict[incomingWarning] = warningArr                    
                    isEndOfError = True

    #process return object
    for key in my_dict:
        arr = my_dict[key]
        for item in arr:
            processreturn = processreturn + ' ' + item
        arryofReturnString.append(f'{key} \n {processreturn}')
    
    #error 
    if isEndOfError:
        arryofReturnString.append(f'<ul><li>{reasonText}</li></ul>\n')

    #process final string
    for arr in arryofReturnString:              
        finalreturn = finalreturn + arr
    print(finalreturn)   
    return finalreturn
    
text = u'Each ice protection components shall be redundant in order to guarantee ice protection functionality after single failure event.'

processText(text)



