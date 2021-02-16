---
title: "Categorizing Grandpa's Memoirs"
author: "Jack Hopper"
date: "1/18/2021"
output:
  html_document:
    keep_md: true
---



## What Is This Project?

My grandfather was a wonderful man. A psychiatrist for nearly 60 years, a WWII veteran, avid car enthusiast and fisherman, father to 8 and grandfather to 28. I consider myself among the luckiest people on earth to have known him. 

My grandfather was a pensive man, always reading, thinking, exclaiming, and waxing poetic about the things he was passionate about -- which is just about everything. Over the course of his career, he would set aside times to write about his thoughts. He had too many thoughts constantly swirling to stick to one subject for long, so these speeches would occasionally turn into rambles. After he passed in early 2020, my family began organizing his belongings and sorting through the stacks of papers he had strewn across the library. While many of his writings were about discrete topics -- the war, growing up in Pittsburgh, the medical profession -- there are some whose topics weren't readily discernable to members of my family who had read them. There are 7 documents here: some of them are recollections of a single day, some are reminiscences on his childhood, and others are about his medical practice.

Naturally, I thought what better way to help everyone out than by doing some text analytics on them?? 

The below is an attempt at summarizing my grandfather's thoughts across seven of his more rambling sessions: what was he talking about, and how does this fit into his life?

#### Getting Started - Working with Text Packages

For this analysis, I leaned heavily on various text analytics packages, namely tidytext, topicmodels, and tesseract. I also need the pdftools package to convert his writings, stored as PDFs, into pictures and then into the R environment.

```
## -- Attaching packages --------------------------------------- tidyverse 1.3.0 --
```

```
## v ggplot2 3.3.2     v purrr   0.3.4
## v tibble  3.0.3     v dplyr   1.0.2
## v tidyr   1.1.1     v stringr 1.4.0
## v readr   1.4.0     v forcats 0.5.0
```

```
## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
## x dplyr::filter() masks stats::filter()
## x dplyr::lag()    masks stats::lag()
```

```
## Using poppler version 0.73.0
```



#### Loading and Cleaning the Data

Since these works are stored as PDFs, I first had to convert everything to an image where tesseract can read the words with the OCR engine. These 7 documents span 26 pages.

I haven't included the code to load in the data here because it is quite repetitive: I know there is a better way to programatically read in chunks of code, but I wanted to focus on the analysis, plus it wasn't too bad writing it out since it is largely cut/paste.



## Topic Modeling
The first thing we can do with this analysis is topic modeling. Topic modeling is an unstructured analytics technique attempts to identify the topics that best describe the information within a collection of documents. The approach I will use here is LDA (Latent Dirichlet Allocation), which is a probabilistic model that categorizes observations across a family of continuous multivariate probability distributions based on the values we select of a few parameters. In other words, with a little guidance on how many topics to include and how "fuzzy" we want each document to be clustered, this algorithm identifies the most likely topics for each document.

##### Building the dataset

Before building the model, though, we must pull these documents into a Document Term Matrix (DTM) - a long table containing 0/1 values to words assigned to them.

```r
#First, we need compile all 7 documents into one, and convert it to a DTM
documents <- tibble(rambling1, rambling2, rambling3, rambling4, rambling5, rambling6, rambling7) %>% 
  gather() %>% 
  mutate(document = 1:7) %>% 
  rename(text = value) %>% 
  select(-key) %>% 
  relocate(document)
```

Before finishing the cleaning process, we should remove stop words -- 1,100+ common words ("a", "the") and non-meaningful words -- and also the word "page" since every page in the documents contains the word 'page' at the bottom. 

```r
#Before converting to a DTM, let's add an extra stop word - the word "page" comes up a lot so let's remove that
stop_words <-
  bind_rows(tibble(
    word = c("page", "1", "2", "3", "4"),
    lexicon = c("custom")
  ), stop_words)
```

Now we can convert to a DTM.

```r
#Convert to DTM
corpus_dtm <- documents %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words, by = "word") %>%
  count(document, word) %>%
  ungroup() %>%
  cast_dtm(document, word, n)
```

#####Building the model

While there are variations on this method and automated ways to select each parameter, those go beyond the scope of this analysis (and, to be honest, my own comprehension of them!). So instead, I have selected the following parameters that will help me identify topics:
1) k - the number of topics. I chose 4 here
2) alpha = the proportion of each document belonging to a topic. If alpha is high (0.5), the document will belong to multiple topics; if alpha is low (0.1), the algorithm will be more confident in its assessment, so each document will likely belong to only one topic
3) Delta - how likely a given word is to belong to a topic

The trick with LDA is that there is no 'optimal' way to assign the number of topics or parameters here. It's an inherently subjective art! I have chosen a k of 4, with alpha = .25 and delta = .1 . This led to the best (most coherent) results of the multiple methods I tried.


```r
corpus_lda <- corpus_dtm %>%
  LDA(k = 4, method = "Gibbs", control = list(alpha = 0.25, delta = 0.1, seed = 1))
```

Now we can analyze the results of the model.
First, let's check out 'beta' which is the topics the model has generated for the whole set.

```r
corpus_topics <- corpus_lda %>%
  tidy(matrix = "beta")

#Visualize the results
corpus_topics %>%
  mutate(topic = str_c("topic", topic)) %>%
  group_by(topic) %>%
  top_n(8, beta) %>%
  ungroup() %>%
  arrange(topic, -beta) %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(mapping = aes(x = term, y = beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered()
```

![](RSB-Memoir-Classification_files/figure-html/unnamed-chunk-8-1.png)<!-- -->
It looks like we can see that of the four topics, one clearly stands out about my grandfather's home life. The 'al' term here is a reference to his best friend Al, with whom he explored much of Italy during his medical training. The others are a bit more fuzzy, but topic2 seems to be about home life, topic3 is about the psychiatry of women, and topic4 is about psychiatry in general.

Now that we have a better sense of what each topic is, we can rename them to better suit their true topics.

```r
corpus_topics %>%
  mutate(topic = str_c("topic", topic)) %>%
  mutate(topic = ifelse(topic == 'topic1', 'Italy & Uncle Al',
                        ifelse(topic == 'topic2', 'Home',
                               ifelse(topic == 'topic3', 'Female Psychiatry', 'Psychiatry')))) %>% 
  group_by(topic) %>%
  top_n(8, beta) %>%
  ungroup() %>%
  arrange(topic, -beta) %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(mapping = aes(x = term, y = beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered()
```

![](RSB-Memoir-Classification_files/figure-html/unnamed-chunk-9-1.png)<!-- -->
Now, we can assign each topic to a document. The 'gamma' term in the LDA model we built contains probabilities that each document belongs to a given topic. We will view those here.

```r
#Topics for each document - gamma
corpus_documents <- corpus_lda %>%
  tidy(matrix = "gamma")

corpus_documents
```

```
## # A tibble: 28 x 3
##    document topic   gamma
##    <chr>    <int>   <dbl>
##  1 1            1 0.0138 
##  2 2            1 0.00157
##  3 3            1 0.0559 
##  4 4            1 0.739  
##  5 5            1 0.00710
##  6 6            1 0.00385
##  7 7            1 0.00918
##  8 1            2 0.983  
##  9 2            2 0.0896 
## 10 3            2 0.238  
## # ... with 18 more rows
```

```r
corpus_documents <- corpus_documents %>%
  mutate(topic = str_c("topic", topic)) %>%
  spread(topic, gamma) %>%
  rename('Italy & Uncle Al' = 'topic1',
         'Home ' = 'topic2',
         'Female Psychiatry' = 'topic3',
         'Psychiatry' = 'topic4') %>% 
  arrange(document)

knitr::kable(corpus_documents, format = 'html', digits = 2)
```

<table>
 <thead>
  <tr>
   <th style="text-align:left;"> document </th>
   <th style="text-align:right;"> Italy &amp; Uncle Al </th>
   <th style="text-align:right;"> Home  </th>
   <th style="text-align:right;"> Female Psychiatry </th>
   <th style="text-align:right;"> Psychiatry </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> 1 </td>
   <td style="text-align:right;"> 0.01 </td>
   <td style="text-align:right;"> 0.98 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 0.00 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 2 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 0.09 </td>
   <td style="text-align:right;"> 0.69 </td>
   <td style="text-align:right;"> 0.22 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 3 </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0.24 </td>
   <td style="text-align:right;"> 0.35 </td>
   <td style="text-align:right;"> 0.35 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 4 </td>
   <td style="text-align:right;"> 0.74 </td>
   <td style="text-align:right;"> 0.22 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 0.04 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 5 </td>
   <td style="text-align:right;"> 0.01 </td>
   <td style="text-align:right;"> 0.04 </td>
   <td style="text-align:right;"> 0.94 </td>
   <td style="text-align:right;"> 0.01 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 6 </td>
   <td style="text-align:right;"> 0.00 </td>
   <td style="text-align:right;"> 0.83 </td>
   <td style="text-align:right;"> 0.06 </td>
   <td style="text-align:right;"> 0.11 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> 7 </td>
   <td style="text-align:right;"> 0.01 </td>
   <td style="text-align:right;"> 0.05 </td>
   <td style="text-align:right;"> 0.10 </td>
   <td style="text-align:right;"> 0.84 </td>
  </tr>
</tbody>
</table>
So there we have it:
Document 1 is about growing up in Pittsburgh;
Document 2 is about Female Psychiatry
DOcument 3 is about Psychiatry in general (note the divergence between general psychiatry and home life)
Document 4 is about Italy & Uncle Al
Document 5 is about Female Psychiatry
Document 6 is about home life
And Document 7 is about psychiatry

That's all! Thanks for reading. I enjoyed this exercise, as it helped me understand what my grandfather often thought about (and gave me a chance to flex my text analytics muscles). 
