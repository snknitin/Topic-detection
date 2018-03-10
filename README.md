# Topic-detection

We would like to understand what common topics (i.e. themes) have been discussed in the conversations. This knowledge should help someone like the ‘Ubuntu team’ to improve their
future versions of Ubuntu.

## Dataset

Ubuntu dataset is a large dataset containing unstructured multi-turn chat dialogues between users and tech supports for Ubuntu OS related issues.
    
    Use the Ubuntu dialog data located at http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/
    Focus on the dialogs/4 folder
There should be ~270K .tsv files, each representing one conversation by different participants. This is the subset we would like to use for this project.


## Methods used

- Latent Dirichlet Allocation
- Non-negative Matrix Factorization

    LDA is based on probabilistic graphical modeling while NMF relies on linear algebra.

Both algorithms take as input a bag of words matrix (i.e., each document represented as a row, with each columns containing the count of words in the corpus)
The aim of each algorithm is then to produce 2 smaller matrices;
     
     A document to topic matrix and a word to topic matrix that when multiplied together reproduce the bag of words matrix with the lowest error.A tf-idf transformer is applied to the bag of words matrix that NMF must process with the TfidfVectorizer.LDA on the other hand, being a probabilistic graphical model (i.e. dealing with probabilities) only requires raw counts, so a CountVectorizer is used.
Stop words are removed and the number of terms included in the bag of words matrix is restricted to vocab_size.

## Problems to solve

- Identify top 10 most popular topics
- Classifier (or topic detector)


## Preprocessing

    2004-09-25T11:29:00.000Z	jblack		geesh. nobody tells me anything.
    2004-09-25T11:30:00.000Z	jblack		Ok. Thanks.
    2004-09-25T11:31:00.000Z	jblack		Got it.
    2004-09-25T11:31:00.000Z	lifeless	jblack	its in the email that jdub sent.. and on the nonameyet wiki

This is the raw data which is converted into a file with the same name in a finished files path

    geesh. nobody tells me anything.Ok. Thanks.Got it.its in the email that jdub sent.. and on the nonameyet wiki

by extacting only the last part of each sentence whic involves the conversation. Since we are identifying topics, user handles and time stamps are irrelevant data.

## Running the code

    python runner.py  <insert data path> <insert finished files path> mode numofTopics vocab_size True/False

- Mode is one of LDA or NMF
- numOfTOpics is set to 10
- vocab_size = 50000
- isProcessed is a boolean value which indicates that data has been preprocessed and can be used directly

## Results

Displaying 10 words per topic.

- For LDA

Seconds for LDA: 517.631
    
    Topic 0:
    ubuntu 10 04 com install http version update upgrade bit
    Topic 1:
    help problem root card work user just ubuntu sound working
    Topic 2:
    install apt sudo command package file add remove installed run
    Topic 3:
    network ubuntu use connect don internet paste wireless connection kernel
    Topic 4:
    ubuntu boot cd windows install drive partition grub mount just
    Topic 5:
    ubuntu know help linux question just need does don use
    Topic 6:
    gnome default firefox desktop flash click know xorg just start
    Topic 7:
    nvidia old kde ubuntu ati ram drivers xubuntu cpu join
    Topic 8:
    file change files screen folder home directory dvd login want
    Topic 9:
    like good use thing just linux getting thats know software

- For NMF

Seconds for NMF: 54.560
    
    Topic 0:
    know just does like linux work using try good use
    Topic 1:
    paste use don http com flood punctuation enter www org
    Topic 2:
    install apt package sudo packages update sources synaptic remove cache
    Topic 3:
    ubuntu channel com support join server offtopic debian https version
    Topic 4:
    windows boot grub cd partition drive live install mount usb
    Topic 5:
    help need ask question problem hi hello channel community https
    Topic 6:
    root sudo password user su login set account passwd change
    Topic 7:
    10 04 upgrade 11 update 12 version release lts manager
    Topic 8:
    gnome kde desktop kubuntu panel terminal menu window login manager
    Topic 9:
    file command files terminal directory open folder line run bin


