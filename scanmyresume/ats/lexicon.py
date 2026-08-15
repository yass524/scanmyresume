# scanmyresume/ats/lexicon.py
import re

STOPWORDS = {
    "a","an","the","and","or","but","if","then","with","for","to","of","in","on","by","from","at","as",
    "is","are","was","were","be","been","being","this","that","these","those","it","its","we","our","you",
    "i","they","their","he","she","them","your","my","me"
}

ACTION_VERBS = {
    "built","implemented","optimized","created","designed","developed","deployed","launched","engineered",
    "constructed","assembled","programmed","coded","automated","refactored","configured","integrated","tested",
    "validated","modeled","simulated","calibrated","commissioned","debugged","diagnosed","repaired","tuned",
    "maintained","led","spearheaded","orchestrated","coordinated","directed","supervised","oversaw","managed",
    "executed","initiated","organized","owned","facilitated","mentored","trained","supported","guided",
    "delivered","achieved","exceeded","improved","enhanced","expanded","scaled","increased","reduced",
    "streamlined","upgraded","resolved","strengthened","surpassed","won","earned","architected","conceived",
    "devised","formulated","innovated","pioneered","strategized","introduced","proposed","instituted",
    "transformed","modernized","collaborated","partnered","contributed","consulted","advised","presented",
    "communicated","drafted","documented","reviewed","negotiated","liaised","aligned","engaged","shared",
    "informed","advocated","facilitated"
}

BULLET_MARKERS = {"•", "-", "–", "—", "*"}

# shared regexes
RE_EMAIL  = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
RE_PHONE  = re.compile(r"(\+?\d[\d\-\s()]{6,}\d)")
RE_URL    = re.compile(r"(https?://|www\.)\S+", re.I)
RE_DIGIT  = re.compile(r"\d")
RE_PCT    = re.compile(r"%")
RE_CURRENCY = re.compile(r"[$€£]|EGP|USD|EUR|GBP", re.I)
