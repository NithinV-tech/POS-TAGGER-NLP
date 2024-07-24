# POS-TAGGER-NLP

 Part-of-Speech Tagger

- As i was getting some error, i have not saved the model. instead will have to train each time.
- However, training the model is not taking much time. 
- I have also added 2 trained models in .pt format in the zip.
     1. LSTM best_model Weights.pt
     2. POSModel_weights.pt


How to run?

    1. pos_tagger is run by postagger.py <option>
    2.  Upon typing -r/-f , wait for a minute and system will prompt you to enter a sentence
    3. upon entering the sentence, it will predict the word and corresponding tag.

Graphs.
 - All the graphs are attached along with the assignment.
 - However, on running the model, it would again generate the same graph


 Preprocessing

- all the words appearing less than 3 times have been given OOV tag
- The dev set had a SYM tag other than the given 12 tags. That is also taken into consideration while building the model.
- Glove embedding has been used and it is stored locally in my system. It is not submitted in the zip file. However i am giving here the link of embedding used : 'https://drive.google.com/uc?id=1R2_AE4lfgn2DsW9sGm2yuImw-6MZNqaG'
- An extra ['PAD'] will be seen in RNN confusion matrix. it can be ignored
