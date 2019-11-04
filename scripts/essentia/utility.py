import spacy

nlp = spacy.load("en_core_web_lg")


def merge_dics(x, y):
    z = x.copy()
    z.update(y)
    return z


def find_dobj_phrase(sent_doc, verb_idx):
    verb_tk = sent_doc[verb_idx]
    if verb_tk.pos_ != "VERB":
        #print("Not a verb, but a {}!".format(verb_tk.pos_))
        return ""

    for tk in verb_tk.children:
        #print("token: {}, dep_: {}".format(tk.text, tk.dep_))
        if tk.dep_ == 'dobj':
            dobj_idx = tk.i
            vp = sent_doc[verb_idx:dobj_idx+1].text            
            #print("Verb phrase found: {}".format(vp))
            return vp

    #print("No dobj found for verb: {} in sent: {}".format(verb_tk.text, sent_doc.text))
    return None

def find_prep_pobj_phrase(sent_doc, verb_idx):
    verb_tk = sent_doc[verb_idx]
    if verb_tk.pos_ != "VERB":
        #print("Not a verb, but a {}!".format(verb_tk.pos_))
        return ""

    for tk in verb_tk.children:
        #print("token: {}, dep_: {}".format(tk.text, tk.dep_))
        if tk.dep_ == 'prep':
            #print("prep tk: {}".format(tk.text))
            for child in tk.children:
                #print("child: {}, dep_: {}".format(child.text, child.dep_))
                if child.dep_ == 'pobj':
                    vp = sent_doc[verb_idx:child.i+1].text            
                    #print("Verb phrase with prep found: {}".format(vp))
                    return vp

    #print("No prep+pobj found for verb: {} in sent: {}".format(verb_tk.text, sent_doc.text))
    return None
    

def test_dobj_phrase():
    sent = u"We want to book a table at Seven Hills for sunset"
    sent_doc = nlp(sent)
    find_dobj_phrase(sent_doc, 3)
    

if __name__ == "__main__":
    test_dobj_phrase()
