# **EMNLP 6-10 Dec 2023**
Shirish Hirekodi

EMNLP main site: https://2023.emnlp.org/  
Papers on ACLAnthology: https://aclanthology.org/events/emnlp-2023/  

## Contents
[Zettels](#zettels)
[Workshops Day 6-DEC-2023 WE](#workshops-day-6-dec-2023-we)
[Tutorials Day 7-DEC-2023 TH](#tutorials-day-7-dec-2023-th)
[Day 1 Main Conf 8-DEC-2023 FR](#day-1-main-conf-8-dec-2023-fr)
[Day 2 Main Conf 9-DEC-2023 SA](#day-2-main-conf-9-dec-2023-sa)
[Day 3 Main Conf 10-DEC-2023 SU](#day-3-main-conf-10-dec-2023-su)
[Poster Stalls](#poster-stalls)

## Zettels

### Z-1 Symbols and DL
* We have reached the limits of what LLM can do by themselves. They need a method to guide them towards an end objective. Could we create a hybrid of symbolism and connectionism which does this? Ref D1-1
* Reinforcement Learning (a)

### Z-2 Models

A mathematical system to find patterns in data and then make predictions

* DPR (a)
* DrKit (b)
* DrFact \(c\)
* BERT (d)
* GPT (e)
* T5 (f)
* LLaMA (g)
* BLOOMZ & mT0 (h)
* LAPDOG (i)
* ESCHER (j)
* CONFLATOR (k)
* Conformer (l)
* MADNET (m)
* Helsinki-NLP/opus-mt-en-de (n)
* Crystal (o)
* Transformers \(p\)
* CNN (q)
* MLP \(r\)

### Z-1a Reinforcement Learning

* MAPPO (1)
* CBS Planner (2)

### Z-1a1 MAPPO

* PPO with centralized value function inputs as MAPPO (Multi-Agent PPO), and PPO with local inputs for both the policy and value function as IPPO (Independent PPO)
* MAPPO is a policy-gradient algorithm, and therefore updates πθ using GA on the objective function

### Z-1a1 CBS Planner

* Conflict-Based Search (CBS) is a popular multi-agent path finding (MAPF) solver that employs a low-level single agent planner and a high-level constraint tree to resolve conflicts
* Multi-Agent Path Finding (MAPF) is the problem of computing collision-free paths for a team of agents in a known environment while minimizing a measure of their travel times

### Z-2a DPR

* Dense Passage Retrieval (DPR) is a set of tools and models for state-of-the-art open-domain Q&A research

### Z-2b DrKit

* DrKIT answers questions based on a virtual KB (VKB) constructed automatically from a text corpus Dhingra et al. (2020)

### Z-2c DrFact

* An efficient Differentiable model for multi-hop Reasoning over knowl- edge Facts
* OpenCSR, Ref: Z-6b

### Z-2d BERT

* Bidirectional Encoder Representations from Transformer
* SOTA on Question Answering (SQuAD v1.1), Natural Language Inference (MNLI)
* gaBERT (1)
* RoBERTa (2)
* mBERT (3)
* XML-R (4)
* MuRIL (5)
* Flan (6)
* BALM (7)
* BART (8)
* BEiT (9)
* DEiT (10)

### Z-2d/1 BERT

* ViT (a)

### Z-2d1 gaBERT

* Model for the Irish language

### Z-2d2 RoBERTa

* Builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates

### Z-2d3 mBERT

* Multilingual BERT (104 langs)
* bert-base-multilingual-cased (a)
* Ref: LRP2, Z-8i1

### Z-2d4 XML-R

* From BERT
* Cross-lingual Language Model Pretraining
* R for RoBERTa, XML-RoBERTa
* 100 languages
* 2.5TB data
* MLM

### Z-2d5 MuRIL

* Multilingual Representations for Indian Languages
* 17 Indian languages and their transliterated counterparts
* BERT based but outperforms mBERT

### Z-2e GPT

* Generative pre-trained transformers
* GPT-2 (1)
* GPT-3 (2)
* GPT-3.5 (3)
* GPT-4 (4)
* PycodeGPT (5)
* ChatGPT (6)
* InstructGPT (7)
* DialoGPT (8)

### Z-2d3a bert-base-multilingual-cased

* Top 104 languages with the largest Wikipedia using a masked language modeling (MLM) objective

### Z-2e1 GPT-2

* GPT-2 was pre-trained on BookCorpus, trained on a dataset of 8 million web pages
* 1.5B
* Recurrence and convolution based architecture

### Z-2e2 GPT-3

* 175B
* Strong "zero-shot" and "few-shot" learning abilities
* Attention based architecture
* XGLM (a)

### Z-2e2a XGLM

* A multilingual version of GPT-3
* Multilingual autoregressive language models on a balanced corpus covering a diverse set of languages
* Largest model is 7.5B
* 30 languages to various extents: English, Russian, Chinese, German, Spanish, French, Japanese, Italian, Portuguese, Greek (modern), Korean, Finnish, Indonesian, Turkish, Arabic, Vietnamese, Thai, Bulgarian, Catalan, Hindi, Estonian, Bengali, Bangla, Tamil, Urdu, Swahili, Telugu, Basque, Burmese, Haitian, Haitian Creole, Quechua

### Z-2e3 GPT-3.5

* 1.3B
* RLHF
* Logic-LM (a)

### Z-2e3a Logic-LM

* Integrates LLMs with symbolic solvers to improve logical problem-solving
* First LLM translates a natural language problem into a symbolic
formulation and then a deterministic symbolic solver performs inference on the formulated problem

### Z-2e4 GPT-4

* 1.76T
* Logic-LM, Ref: Z-2e3a

### Z-2e5 PycodeGPT

* 110M
* Based on GPT-Neo
* Vocabulary size of 32K

### Z-2f T5

* C4 dataset
* For multiple NLP tasks
* Code T5 (1)
* CONTRASTE-MTL (2)
* TRUE (3)

### Z-2e6 ChatGPT

* ChatBot with prompt as context
* Either 3.5 or 4

### Z-2e7 InstructGPT

* RLHF
* From GPT 3

### Z-2e8 DialoGPT

* From GPT 2
* Trained on 147M conversation-like exchanges extracted from Reddit
* Create a chat bot in just 10 lines of code as shown on model card

### Z-2g LLaMA

* Large Language Model Meta AI
* Alpaca (1)
* Vicuna (2)

### Z-2f1 Code T5

* Denoised seq2seq
* For programming languages

### Z-2f2 CONTRASTE-MTL

* CONTRastive learning to enhance the
* ASTE performance
* Aspect Sentiment Triplet Extraction (ASTE)
* ASTE is the most interpretable Aspect-based Sentiment Analysis (ABSA) task
* Train T5 for ABSA tasks

### Z-2f3 TRUE

* Predict whether a generated text is factually consistent with respect to a grounding text 
* Fine-tuning T5-11B on the Adversarial NLI

### Z-2g1 Alpaca

* Fine-tuned from the LLaMA 7B model on 52K instruction-following demonstrations
* Behaves qualitatively similarly to text-davinci-003
* Surprisingly small and easy/cheap to reproduce  at < $600
* Intended only for academic research and any commercial use is prohibited

### Z-2g2 Vicuna

* Family of open-source language models instruct tuned from LLaMA
$300
* Trained on ShareGPT
* Vicuna-13B achieves more than 90% quality of OpenAI ChatGPT and Google Bard (subjective, fun eval)

### Z-2h BLOOMZ & mT0

* Multitask prompted finetuning (MTF)
* Apply MTF to the pretrained multilingual BLOOM and mT5
* Following human instructions in dozens of languages zero-shot
* Ref: LRP2, Z-8i1

### Z-2d6 Flan

* Based on BERT
* Fine tuning is done using the “Flan” prompt tuning and dataset collection
* Flan-T5-<size>
* Flan-UL2

### Z-2j ESCHER

* Barba et al., 2021
* Reframing WSD as a span extrac- tion problem is Extractive Sense Comprehension (ESC)
* A transformer-based neural architec- ture
* Few-shot
* Combine data annotated with senses from different lexical resources

### Z-2k CONFLATOR

* Hindrance in the adoption of SoTA Transformer-based LMs for code-mixing can be at- tributed to data scarcity
* 6-headed MHA architecture

### Z-2l Conformer

* Combine CNN and Transformer
* SR applications

### Z-2d7 BALM

* BERT based
* Framework incorporates monolingual priors into an MT pipeline ie simple MT translates En to De without knowing anything about the two languages

### Z-2d8 BART

* Denoising Autoencoder from Transformer
* Trained to reconstruct the original sentence from a corrupted version of it, which is called denoising autoencoding
* Learn more robust representations of the text and to handle more complex language tasks
* BERT task = MLM
* BART task = denoising autoencoding
* MBart-50 (a)

### Z-2d8a MBart-50

* Slightly different from mBART
* Language id token is used as a prefix for both source and target text
* Has its own tokenizer MBart50Tokenizer
* Multilingual

### Z-2d9 BEiT

* Bidirectional Encoder representation from Image Transformers
BERT like, Transformer based
* Masked image modeling task to pretrain vision

### Z-2d10 DEiT

* Data-efficient image Transformers
* No convolutional layer can achieve competitive results against SOTA on ImageNet

### Z-2d/1 BERT

* ViT (a)
* BanglaBERT (b)
* CLIP \(c\)

### Z-2i LAPDOG

* Retrieval side: Transformer based
* Generator side: Fusion-in-decoder technique

### Z-2m MADNET

* Maximizes addressee deduction expectation in heterogeneous graph neural networks for Multi Party Conversation generation

### Z-2n Helsinki-NLP/opus-mt-en-de

* For translation and text-to-text generation
* MT

### Z-2o Crystal

* Introspective reasoning model commonsense QA

### Z-2p Transformers

Network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely

* Swin (1)
* NLLB (2)
* M2M-100 (3)
* Longformer (4)
* CORECT (5)
* PaLM (6)
* ChatGLM (7)
* Galactica (8)
* LaMDA (9)
* MiniLM (10)

### Z-2d/1a ViT

* Applying a standard Transformer directly to images with the fewest possible modifications
* Image patches are treated the same way as tokens

### Z-2d/1b BanglaBERT

* Low-resource
* ELECTRA setup and objective which is pretrained with Replaced Token Detection (RTD) objective

### Z-2d/1c CLIP

* Contrastive Language-Image Pre-training
* Simplified version of ConVIRT trained from scratch
* 400 million (image, text) pairs
* Different from MLM, CLIP learns image and text representations through image-text contrastive (ITC) pretraining

### Z-2p1 Swin

* A hierarchical Transformer whose representation is computed with Shifted WINdows
* Limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection

### Z-2p2 NLLB

* No Language Left Behind
* 202 languages
* 54.5B Sparsely Gated Mixture-of-Experts model
* 3.3B and 1.3B Dense Transformer models
* 1.3B and 600M Dense transformer models
* Low-Resource

### Z-2p3 M2M-100

* Z-2p5 CORECT
* Z-2p6 PaLM
* COnversation understanding model
* using RElational Temporal Graph Neural Network with Auxiliary Cross-Modality Interaction
* Transformer based
* Capture conversation-level cross-modality interactions and utterance-level temporal dependencies with the modality-specific manner for conversation understanding
* 540-billion parameter, densely activated, decode only, Transformer LM
* See PaLM note in Z-2p9
* Single model that could generalize across domains and tasks
* “Pathways” a Google Project will enable training a single model to do thousands or millions of things
* Pathways Language Model
* Flan-PaLM (a)
* Minerva (b)

### Z-2p6a Flan-PaLM

* A PaLM model trained on the Flan dataset
* Ref: Z-2d6

### Z-2p6b Minerva

* Quantitative reasoning tasks. This model is able to process scientific and mathematical questions formulated in natural language,
and generate step-by-step solutions using correct LATEX notation
* Trained on scientific and mathematical data
* Code generation, Formal math, LM applied to formal and synthetic math problems

### Z-2p7 ChatGLM

* Based on GLM architecture
* Transformer-based
* English and Chinese
* 130B

### Z-2p8 Galactica

* Transformer architecture in a decoder-only setup
* Performs well on reasoning, on mathematical MMLU, and on MATH
* galactica.org
* 48 million papers, textbooks and lecture notes, millions of compounds and proteins, scientific websites, encyclopedias and more

### Z-2p9 LaMDA

* LM for Dialogue Applications
* Google
* LaMDA evolved into PaLM and PaLM became Gemini
* Decoder-only transformer LM
* Bard (a)

### Z-2p9a Bard

* Conversational generative AI chatbot

### Z-2p10 MiniLM

* Knowledge distillation is the process of compressing the knowledge of a large model (teacher) into a smaller one (student)
* MiniLM proposes novel approaches to perform distillation of large models like BERT and RoBERTa into smaller models that could be 99% accurate on certain tasks while being more than 2 times faster in inference
* v2 generalizes deep self-attention distillation in MiniLMv1 by using self-attention relation distillation for task-agnostic compression of pre-trained Transformers.

### Z-2q CNN

* A regularized type of feed-forward neural network that learns feature engineering by itself via filters (or kernel) optimization
* VGG16 (1)

### Z-2q1 VGG16

* Also called VGGNet
* 16 layer
* Improves on AlexNet
* Smaller filters (3x3) instead of larger

### Z-2r MLP

* Typical feedforward nn
* Differentiate data that is not linearly separable

### Z-3 Dataset

A collection of inputs used to train a model to learn patterns
* ARC-Open (a)
* OBQA (b)
* Psycon \(c\)
* Wow (d)
* DD (e)
* MS-MARCO (f)
* MATRES (g)
* Gigaword (h)
* BoolQ (i)
* WebNLG (j)
* DBpedia Triplets (k)
* OneCommon (l)
* TopV2 (m)
* TextEditing (n)
* ATIS (o)
* MultiAtis++ \(p\)
* MultiWOZ (q)
* COPIOUS \(r\)
* TONGUESWITCHER (s)
* Bangor Miami (t)
* Fisher (u)
* CoVoST (v)
* MuST-C (w)
* SEA (x)
* ICON (y)
* LibriSpeech (z)

### Z-3/1 Dataset

* Multilingual LibriSpeech (a)
* VoxPopuli (b)
* ULCA \(c\)
* MUCS (d)
* FLEURS (e)
* SEAME (f)
* ConvAI (g)
* ROCStory (h)
* PersonaChat (i)
* PersuasionForGood (j)
* Ubuntu IRC (k)
* SODA (l)
* ASSET (m)
* WritingPrompts (n)
* DailyDialog++ (o)
* COSMOS QA \(p\)
* Swag (q)
* DuoRC \(r\)
* OpenBookQA (s)
* ARC (t)
* COMMONSENSEQA (u)
* STAC (v)
* Molweni (w)
* GSM8K (x)
* MultiArith (y)
* ASDiv (z)

### Z-3/2 Dataset

* SVAMP (a)
* WMT-14 (b)
* Flan Ref: Z-2d6
* SNLI \(c\)
* MNLI (d)
* CapitolWords (e)
* IEMOCAP (f)
* CMU-MOSEI (g)
* DNIC (h)
* XL-WSD (i)
* BBC (j)
* arXiv (k)
* MLW_data (l)
* dom_project (m)
* JOKER@CLEF 2022 (n)
* PAWS (o)
* MARC \(p\)
* COPA (q)
* StoryCloze \(r\)
* EXAMS (s)
* OPUS (t)
* MixMT (u)
* SAMsum (v)
* LinCE (w)
* MixSentiment Malayalam (x)
* MixSentiment Tamil (y)
* BanglaAbuseMeme (z)

### Z-3/3 Dataset

* QuAC (a)
* CoQA (b)
* DoQA \(c\)
* QASC (d)
* McRae (e)
* CSLB (f)
* IndoMMLU (g)
* NusaX (h)
* ASQA (i)
* QAMPARI (j)
* ELI5 (k)
* People-Diversity (l)
* Cultural-Diversity (m)
* MDC (n)
* IGLU (o)
* HPD \(p\)
* HotpotQA (q)
* PubMedQA \(r\)
* MedMCQA (s)
* Logic/Reasoning (t)

### Z-3a ARC-Open

* AI2’s Reasoning Challenge (ARC) dataset is a multiple-choice question-answering dataset, containing questions from science exams from grade 3 to grade 9
* Easy and Challenge

### Z-3b OBQA

* OpenBookQA Dataset
* Contains questions that require multi-step reasoning, use of additional common and commonsense knowledge, and rich text comprehension

### Z-3c Psycon

* Conversational dataset for psychotherapy
* dialogue-level - (gender, age, persona)
* utterance-level - user’s sentiment and therapist’s politeness, and interpersonal behaviour

### Z-3d Wow

* Wizard of Wikipedia
* Train and evaluate dialogue systems for knowledgeable open dialogue with clear grounding

### Z-3e DD

* Daily Dialog
* Mxulti-turn open-domain English dialog dataset
* Around 8 speaker turns per dialogue with around 15 tokens per turn

### Z-3f MS-MARCO

* Microsoft MAchine Reading Comprehension
* Collection
* QA, NLG, passage ranking, keyphrase extraction, crawling

### Z-3g MATRES

* Multi-Axis Temporal RElations for Start-points
* Capture the temporal structure of events

### Z-3h Gigaword

* Headline-generation on a corpus of article pairs
* Summarization, where given a document, the goal is to predict its summary

### Z-3i BoolQ

* yes/no questions
* triplet of (question, passage, answer)
* NLI application

### Z-3j WebNLG

* Data/Text pairs where the data is a set of triples extracted from DBpedia

### Z-3k DBpedia Triplets

* DBpedia KB in RDF
* Subject-Object-Predicate

### Z-3l OneCommon

* 6,760 dialogues

### Z-3m TopV2

* Task Oriented Parsing v2
* Multi-domain task-oriented semantic parsing dataset

### Z-3n TextEditing

* DSL, domain specific language
* Text editing without the need of understanding the grammar and semantics of regular expressions, conditionals, loops, etc

### Z-3o ATIS

* Audio and transcripts about humans asking for flight information
* English, Hindi, Turkish
* Ref: Z-3p

### Z-3p MultiAtis++

* Extends ATIS to 6 more languages, and covers a total of 9 languages, that is, English, Spanish, German, French, Portuguese, Chinese, Japanese, Hindi and Turkish
* Ref: Z-3o

### Z-3q MultiWOZ

* Multi-Domain Wizard-of-Oz
* Task-Oriented Dialogue Modelling

### Z-3r COPIOUS

* Biodiversity entities
* Categories: taxon names, geographical locations, habitats, temporal expressions and person names

### Z-3s TONGUESWITCHER

* German-English code-switching
* Language-ambiguous words which can only be resolved in context,
Morphologically mixed words in both English and German morphemes

### Z-3t Bangor Miami

* Code-switched Spanish-English conversation
* Audio and transcripts

### Z-3u Fisher

* Audio in code-switched Spanish and English

### Z-3v CoVoST

* Multilingual and diversified ST dataset based on the Common Voice project
* Includes low resource languages

### Z-3w MuST-C

* Audio of  English TED Talks and corresponding transcriptions and translations for ST

### Z-3x SEA

* Code-switched South East Asian languages
* Closed dataset?

### Z-3y ICON

* Two codemixed data pairs HI-EN and BN-EN are provided for developing sentiment analysis systems
* Closed dataset?

### Z-3/1d MUCS

* 1,000 hours of audioooks (Project Gutenberg)
* Part of the LibriVox project
* Ref: Z-3/1a

### Z-3/1a Multilingual LibriSpeech (MLS)

* 8 languages - English, German, Dutch, Spanish, French, Italian, Portuguese, Polish
* Ref: Z-3z

### Z-3/1b VoxPopuli

* Multilingual speech corpus for representation learning, semi-supervised learning and interpretation
* European Parliament event recordings
* en, de, fr, es, pl, it, ro, hu, cs, nl, fi, hr, sk, sl, et, lt, pt, bg, el, lv, mt, sv or da

### Z-3/1c ULCA

* Indian language datasets and models
* Supported to various extents: Marathi, Hindi, Telugu, Punjabi, Bengali, Gujarati, Malayalam, Tamil, Kannada, Odia, Assamese, Urdu, Bodo, Nepali, Sindhi, Sanskrit, Maithili, Santali, Konkani, Manipuri, Dogri, Goan Konkani, Kashmiri

### Z-3/1d MUCS

* Low-resource multilingual and code-switching (MUCS) ASR
* Transcribed speech data in code-switched Hindi-English and Bengali-English

### Z-3/1e FLEURS

* Few-shot Learning Evaluation of Universal Representations of Speech benchmark
* 102 languages built MT FLoRes-101 benchmark

### Z-3/1f SEAME

* Singapore and Malaysia
* Mandarin and English

### Z-3/1g ConvAI

* Dataset collected during the Conversational Intelligence Challenge (ConvAI), 2017
* 4,000 dialogues; 10 chatbots; 1,000 volunteers
* Versions 1, 2 and 3

### Z-3/1h ROCStory

* Evaluate story understanding and script learning: the ‘Story Cloze Test’
* Choose the correct ending to a four-sentence story

### Z-3/1i PersonaChat

* 164,356 utterances between crowdworkers and act the part of a given provided persona

### Z-3/1j PersuasionForGood

* Dialogues and annotated emerging persuasion strategies
* Personality, morality, value systems, and their willingness for donation

### Z-3/1k Ubuntu IRC

* Logs of Ubuntu-related channels on the Libera.chat IRC network

### Z-3/1l SODA

* English dialogue dataset covering a wide variety of social interactions

### Z-3/1m ASSET

* A Dataset for Tuning and Evaluation of Sentence Simplification Models with Multiple Rewriting Transformations
* Assessing sentence simplification in English
* Each simplification is produced by executing several rewriting transformations

### Z-3/1n WritingPrompts

* From Reddit’s WritingPrompts forum
* Users inspire each other to write by submitting story premises, or prompts, and other users freely respond

### Z-3/1o DailyDialog++

* Consists of (i) five relevant responses for each context and (ii) five adversarially crafted irrelevant responses for each context

### Z-3/1p COSMOS QA

* Problems that require commonsense-based reading comprehension, formulated as multiple-choice questions
* That require reasoning beyond the exact text spans in the context

### Z-3/1q Swag

* Multiple choice questions about a rich spectrum of grounded situations

### Z-3/1r DuoRC

* Collection of 7680 pairs of movie plots where each pair in the collection reflects two versions of the same movie - one from Wikipedia and the other from IMDb
* Reading Comprehension (RC) system
* Low lexical overlap between questions and their corresponding passages

### Z-3/1s OpenBookQA

* Modeled after open book exams for assessing human understanding of a subject
combining an open book fact (e.g., metals conduct electricity) with broad common knowledge (e.g., a suit of armor is made of metal) obtained from other sources

### Z-3/1t ARC

* AI2 Reasoning Challenge (ARC)
* A Challenge Set and an Easy Set
* Natural, grade-school science questions

### Z-3/1u COMMONSENSEQA

* To capture common sense beyond associations, we extract from CONCEPTNET (Speer et al., 2017) multiple target concepts that have the same semantic relation to a single source concept

### Z-3/1v STAC

* Multiparty chats annotated for discourse structure in the style of SDRT
* Provides full discourse structures for multi-party dialogues
* Interleaved threads, creative language, and interactions between linguistic and extra-linguistic contexts

### Z-3/1w Molweni

* Machine reading comprehension (MRC)
* Discourse relations
* Derived from: Ubuntu Chat Corpus

### Z-3/1x GSM8K

* Problems at the grade school math level
* Problems take between 2 and 8 steps to solve
* Using basic arithmetic operations (+ − ×÷)
* High linguistic diversity

### Z-3/1y MultiArith

* Arithmetic Reasoning
* Could not find official dataset

### Z-3/1z ASDiv

* Academia Sinica Diverse MWP Dataset
* Math word problem
* Problem types taught in elementary school

### Z-3/2a SVAMP

* Math word problem
* Carefully chosen variations over examples sampled from existing datasets

### Z-3/1z ASDiv

* Academia Sinica Diverse MWP 

### Z-3/2a SVAMP

* Math word problem

### Z-3/2b WMT-14

* Collection of datasets used in shared tasks of the Ninth Workshop on Statistical Machine Translation
* Four tasks: a news translation task, a quality estimation task, a metrics task, a medical text translation task

### Z-3/2c SNLI

* Stanford Natural Language Inference
* 570k sentence pairs labeled for entailment, contradiction, and semantic independence

### Z-3/2d MNLI

* Multi-Genre Natural Language Inference
* 433k sentence pairs annotated with textual entailment
* Range of genres of spoken and written text, and supports a distinctive cross-genre generalization evaluation
* XNLI (1)
* PAWS-X (2)

### Z-3/2d1 XNLI

* Cross-lingual language understanding (XLU) and low-resource cross-language transfer
* A scalable way to build multilingual systems is through cross-lingual language understanding (XLU), in which a system is trained primarily on data in one language and evaluated on data in others
* XNLI, by extending NLI corpora to 15 languages
* 7500 development and test examples
* English, French, Spanish, German, Greek, Bulgarian, Russian, Turkish, Arabic, Vietnamese, Thai, Chinese, Hindi, Swahili and Urdu
* Ref: Z-3/2d

### Z-3/2e CapitolWords

* 11k speeches given by the main protagonists of the 2016 U.S. Presidential election
* From textacy

### Z-3/2f IEMOCAP

* Interactive emotional dyadic motion capture database
* Emotions are expressed through a combination of verbal and non-verbal channels, a joint analysis of speech and gestures is required to understand expressive human communication
* Text and Video

### Z-3/2g CMU-MOSEI

* CMU Multimodal Opinion Sentiment and Emotion Intensity
* Allows for in-depth studies of multimodal language

### Z-3/2h DNIC

* Diverse Names in Context
* Combine templates with names that are known to be strongly associated with particular social groups
* English, through Spanish, Russian, Arabic and Chinese

### Z-3/2i XL-WSD

* Extra-Large and Cross-Lingual Evaluation Framework for Word Sense Disambiguation
* Cross-lingual evaluation benchmark for the WSD task featuring sense-annotated development and test sets in 18 languages from six different linguistic families
* Language-specific silver training data

### Z-3/2j BBC

* BBC News: 2225 documents, 2004-2005, (business, entertainment, politics, sport, tech)
* BBC Sport: 37 documents, 2004-2005, (athletics, cricket, football, rugby, tennis)

### Z-3/2k arXiv

* arXiv is a preprint repository for scholarly articles in various fields
* Meta data on physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering, systems science, and economics

### Z-3/2l MLW_data

* Misoda Working Group
* Ref: Z-3/2m
* Old Occitan (Old Provençal) character recognition

### Z-3/2m dom_project

* Misoda Working Group
* Ref: Z-3/2l
* Old Occitan (Old Provençal) character recognition

### Z-3/2n JOKER@CLEF 2022

* Conference & Labs of the Evaluation Forum
* Joker Track of CLEF evaluates humour which involves understanding implicit cultural references and/or double meanings, especially in the case of wordplay, which raises not only the question of its (un)translatability, but also how to detect and classify instances of this complex phenomenon

### Z-3/2o PAWS

* Paraphrase Adversaries from Word Scrambling
* 108,463 well-formed paraphrase and non-paraphrase pairs with high lexical overlap
* From sentences in Quora and Wikipedia
* PAWS-X [1]

### Z-3/2o1 PAWS-X

* Cross-lingual Adversarial Dataset for Paraphrase Identification
* 23,659 human translated PAWS evaluation pairs
* English to French, Spanish, German, Chinese, Japanese and Korean

### Z-3/2p MARC

* Multilingual Amazon Reviews Corpus
* Reviews in English, Japanese, German, French, Spanish, and Chinese
* Review text, the review title, the star rating, an anonymized reviewer ID, an anonymized product ID, and the coarse-grained product category (e.g., ‘books’, ‘appliances’, etc.)
* 200,000 reviews in the training set for each of the language
* A language detection algorithm (Bojanowski et al., 2017) to determine the language of the review text

### Z-3/2q COPA

* Choice of Plausible Alternatives
* An Evaluation of Commonsense Causal Reasoning
* One thousand English-language questions
* Each question gives a premise and two plausible causes or effects, where the correct choice is the alternative that is more plausible than the other
* Loosely based on Winograd Schema Challenge
* XCOPA [1]

### Z-3/2q1 XCOPA

* Cross-lingual Choice of Plausible Alternatives
* Multilingual Dataset for Causal Commonsense Reasoning
* CR in 11 languages, which includes resource-poor languages like Eastern Apur´ımac Quechua and Haitian Creole
* CR must bridge between premises and possible hypotheses with world knowledge that is not explicit in text
* Temporal and spatial relations, causality, laws of nature, social conventions, politeness, emotional responses, and multiple modalities

### Z-3/2r StoryCloze

* CR for story understanding, story generation and script learning
* Choose the correct ending to a four sentence story
* 50k five-sentence commonsense stories
* Related to ROCStory, Ref: Z-3/1h
* XStoryCloze [1]

### Z-3/2r1 XStoryCloze

* Professionally translated validation split of the English StoryCloze dataset to 10 other typologically diverse languages
* Ref: Z-3/2r
* English + {Russian, Simplified Chinese, Spanish Latin American, Arabic, Hindi, Indonesian, Telugu, Swedish, Basque, Burmese}

### Z-3/2s EXAMS

* Multi-Subject High School Examinations Dataset for Cross-Lingual and Multilingual QA
* 24,000 high quality high school exam questions in 16 languages, covering 8 language families and 24 school subjects
* Albanian, Arabic, Bulgarian, Croatian, French, German, Hungarian, Italian, Lithuanian, Macedonian, Polish, Portuguese, Serbian, Spanish, Turkish, Vietnamese

### Z-3/2t OPUS

* Open Parallel Corpus
* Growing collection of translated texts from the web
* Convert and align data with added linguistic annotation
* 90 languages from several domains
* 3,800 language pairs
* movie subtitles to GNOME documentation to the Bible
* OPUS-100 (1)

### Z-3/2t1 OPUS-100

* Massively multilingual dataset sampled from OPUS
* 55M English-centric sentence pairs covering 100 languages
* Afrikaans, Breton, Welsh, Basque, Western Frisian, Hebrew, Hindi, Japanese, Kannada, Latvian, Marathi, Malay, Norwegian Bokmål, Portuguese, Russian, Tamil, Yiddish, Chinese, etc

### Z-3/2u MixMT

* Code-mixing (or code-switching)
* WMT 2022 shared task
* Two subtasks involving a code-mixed language i.e. Hinglish
[1] Monolingual to code-mixed machine translation (M2CM) [2] Code-mixed to monolingual machine translation (CM2M)
* HinGE training dataset

### Z-3/2v SAMsum

* 16k chat dialogues with manually annotated summaries
* Includes chit-chats, gossiping about friends, arranging meetings, discussing politics, consulting university assignments with colleagues, etc
* The style and register are diversified - conversations could be informal, semi-formal or formal, they may contain slang words, emoticons and typos
* Samsung R&D Institute Poland
* GupShup (1)

### Z-3/2v1 GupShup

* A subset of conversations and summaries from the SAMSum corpus
* 6,800 code-switched (Hi-En) conversations and their corresponding human-annotated summaries in En and Hi-En

### Z-3/2w LinCE

* Linguistic Code-switching Evaluation
* Combines ten corpora covering four different code-switched language pairs
* Spanish-English, Nepali-English, Hindi-English, and Modern Standard Arabic-Egyptian Arabic
* Language identification, named entity recognition, part-of-speech tagging, and sentiment analysis
* [1] multiple language pairs from high- and low-resource languages with a
* reasonable range of CMI [2] typologically-diverse languages [3] variety of NLP tasks including core tasks and downstream applications, and [4] different code-switching domains from social media platforms
* Speakers transliterate Hindi employing mostly ad-hoc phonological rules
* SA task uses SentiMix (SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets). The sentiment labels are positive, negative, or neutral, and the code-mixed languages are English-Hindi and English-Spanish

### Z-3/2x MixSentiment Malayalam

* The Malayalam script is the Vatteluttu alphabet extended with symbols from the Grantha alphabet
* Code-mixed dataset for Malayalam-English
* youtube-comment-scraper tool to download the comments
* 7,743 distinct sentences

### Z-3/2y MixSentiment Tamil

* Code-mixed dataset for Tamil-English
* youtube-comment-scraper tool to download the comments
* 15,744 distinct sentences
* Krippendorff’s alpha (α) (Krippendorff, 1970) measures inter-annotator agreement

### Z-3/2z BanglaAbuseMeme

* Annotations for 4,043 Bengali memes, among which 1,515 memes are abusive
* Lexicon of 69 offensive Bengali terms
* Human annotation

### Z-3/3a QuAC

* Question Answering in Context
* 14K information-seeking QA dialogs
* 100K questions in total
* Model, understand and participate in information seeking dialog
* (1) a student who poses a sequence of freeform questions to learn as much as possible about a hidden Wikipedia text, and (2) a teacher who answers the questions by providing short excerpts (spans) from the text
* QuAC-NH (1)

### Z-3/3a1 QuAC-NH

* Question Answering in Context - Noisy History
* Derived from QuAC, Ref: Z-3/3a
* Contains incorrect answers

### Z-3/3b CoQA

* Conversational Question Answering
* Measure the ability of machines to participate in a question answering style conversation
* First, concerns the nature of the question. Second, is to ensure the naturalness of answers. Third, enable building QA systems that perform robustly across domains
* 127k conversation turns collected from 8k conversations
* Free-form answers
* Seven diverse domains

### Z-3/3c DoQA

* Domain-specific QA
* Accessing domain-specific FAQs via conversational QA
* 2,437 information-seeking question/answer dialogues on three different domains (Cooking, Travel and Movies)
* 10,917 questions in total

### Z-3/3d QASC

* Question Answering via Sentence Composition
* Multi-hop reasoning dataset
* (a) the facts to be composed are annotated in a large corpus, and (b) the decomposition into these facts is not evident from the question itself
* 9,980 8-way multiple-choice questions from elementary and middle school level science, with a focus on fact composition

### Z-3/3e McRae

* McRae Feature Norms dataset
* Collection of experimental data in the field of psycholinguistics and cognitive science
* Lexical semantics and the organization of conceptual knowledge
* Investigate how people organize and represent concepts in their minds

### Z-3/3f CSLB

* Centre for Speech, Language and Brain
* Aim to replicate the McRae norms

### Z-3/3g IndoMMLU

* Indonesian Massive Multitask Language Understanding
* School exams serve as a powerful means to assess the reasoning abilities
* MMLU, Ref: Z-3/3t12
* 64 tasks across different subject areas and education levels in Indo
* 25% of data encompasses nine distinct local languages and cultures in Indonesia, namely Lampungic (ljp), Balinese (ban), Makassarese (mak), Banjarese (bjn), Madurese (mad), Sundanese (sun), Javanese (jav), Dayak Ngaju (nij), and Minangkabau.
* Includes dataset NusaX, Z-3/3h and FactQA, IDK-MRC and TyDiQA

### Z-3/3h NusaX

* Multilingual Parallel Sentiment Dataset for 10 Indonesian Languages
* Low-resource
* Linguistically-diverse with more than 700 languages
* Acehnese, Balinese, Banjarese, Buginese, Madurese, Minangkabau, Javanese, Ngaju, Sundanese, and Toba Batak
* Derived from SmSA an existing Indonesian sentiment analysis dataset containing comments and reviews from the IndoNLU benchmark
* Austronesian language family under the Malayo-Polynesian subgroup

### Z-3/3i ASQA

* Answer Summaries for Questions which are Ambiguous)
* Long-form QA
* Reliable metric for measuring performance on ASQA
* One of the primary data sources is the ELI5 dataset (Fan et al., 2019) that pairs open-ended questions with paragraph-long answers written by users of the “Explain Like I’m Five” Reddit forum
* The AMBIGQA dataset that connects ambiguous factoid questions with disambiguations: pairs of disambiguated questions and unique short answers to these questions
* The CONDITIONALQA task requires systems to identify conditions under which the extracted answers are valid
* High quality long-form answers to 6,316 ambiguous factoid questions

### Z-3/3j QAMPARI

* Questions with many Answers over Multiple Paragraphs, Indeed
* At least 5 answers, with an average of 13 answers
* 2K development and test questions and more than 60K training examples

### Z-3/3k ELI5

* Explain Like I’m Five
* Long Form Question Answering
* 270K threads
* Emphasizes the dual challenges of isolating relevant information within long source documents and generating paragraph-length explanations in response to complex, diverse questions

### Z-3/3l People-Diversity

* People-seeking prompts focused on occupations by hand-crafting
* Consider a (limited) set of sensitive attributes (e.g., Gender, Ethnicity) that we want the LLM response to be diverse towards

### Z-3/3m Cultural-Diversity

* Hand-crafted a set of templates

### Z-3/3n MDC

* Minecraft Dialogue Corpus
* 509 human-human written dialogues, screenshots and complete game logs
* Partly inspired by the HCRC Map Task Corpus (Anderson et al., 1991)
* Koller et al. (2010) design a challenge where systems with access to symbolic world representations and a route planner generate real-time instructions to guide users through a treasure hunt in a virtual 3D world
* Collaborative Building Task (CBT) as a two-player game between an Architect (A) and a Builder (B). A is given a target structure (Target) and has to instruct B via a text chat interface to build a copy of Target on a given build region

### Z-3/3o IGLU

* Interactive Grounded Language Understanding in a Collaborative Environment
* The goal of this competition is to build embodied agents that learn to solve a task while provided with grounded natural language instructions in a collaborative environment
* IGLU competition datasets for RL and NLP
* IGLU-NLP (1)
* IGLU-MULTI (2)

### Z-3/3o1 IGLU-NLP

* 8,136 single-turn data pairs of instructions and actions

### Z-3/3o2 IGLU-MULTI

* Partially motivated by the HCRC Map Task Corpus (Thompson et al., 1993)
* Multi-turn data-collection similar to MDC, Z-3/3n
* 667 dialog- and building turns across 120 game-plays

### Z-3/3p HPD

* Harry Potter Dialogue
* Facilitating the study of Dialogue Agents - Character aligning
* English and Chinese
* 1042 dialogue sessions for training, 149 sessions for testing
* Experimental results show that HPD can help LLMs, such as ChatGPT, better align with the behaviors of Harry Potter

### Z-3/3q HotpotQA

* 113k Wikipedia-base question-answer pairs
* The questions require finding and reasoning over multiple supporting documents to answer
* The questions are diverse and not constrained to any pre-existing knowledge bases or knowledge schemas
* Sentence-level supporting facts required for reasoning, allowing QA systems to reason with strong supervision and explain the predictions
* New type of factoid comparison questions to test QA systems’ ability to extract relevant facts and perform necessary comparison

### Z-3/3q/1 NaturalQuestionsQA

* Real anonymized, aggregated queries issued to the Google search engine
* 307,373 training examples with single annotations
* 7,830 examples with 5-way annotations for development data
* 7,842 examples with 5-way annotated sequestered as test data
* 302 examples with 25-way annotations

### Z-3/3q/2 TriviaQA

* Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension
* 650K question-answer-evidence triples
*  95K question-answer pairs authored by trivia enthusiasts

### Z-3/3q/2 WebQuestions

* Question-answer pairs obtained from non-experts
* Freebase related (FREE917)
* 5,810 questions

### Z-3/3r PubMedQA

* From PubMed abstracts
* PubMed a search engine providing access to over 25 million references of biomedical articles
* Biomedical Research Question Answering
* 1k expert-annotated, 61.2k unlabeled and 211.3k artificially generated QA instances

### Z-3/3s MedMCQA

* Multiple-Choice Question Answering
* Address real world medical entrance exam questions
* 194k high-quality AIIMS & NEET PG entrance exam MCQs covering 2.4k healthcare topics and 21 medical subjects

### Z-3/3t Logic/Reasoning

* PrOntoQA (1)
* ProofWriter (2)
* FOLIO (3)
* BIG-Bench (4)
* AR-LSAT (5)
* T-REx (6)
* Entailment Bank (7)
* ConceptNet (8)
* Microsoft Concept Graph (9)
* WordNet (10)
* GenericsKB (11)
* MMLU (12)
* MATH (13)
* LogiQA (14)
* VQA (15)
* OK-VQA (16)
* BLIP-2 (17)
* PICa (18)

### Z-3/3t1 PrOntoQA

* Proof and Ontology-Generated Question-Answering
* Inspired by the ProofWriter dataset (Tafjord et al., 2021), Ref: Z-3/3t2
* Each example is generated from an ontology and has a unique proof 
* A QA dataset which generates examples with chains-of-thought that describe the reasoning required to answer the questions correctly
* The sentences in the examples are syntactically simple and amenable to semantic parsing

### Z-3/3t2 ProofWriter

* ProofWriter is not a dataset, but a generative model
* PW works with the RuleTaker dataset
* RuleTaker generates datasets of theories and assertions meant to test the logical reasoning capabilities of a model

### Z-3/3t3 FOLIO

* Natural Language Reasoning with First-Order Logic
* Human-annotated, opendomain, and logically complex and diverse dataset for reasoning in NL with FOL annotations
* 1,435 examples (unique conclusions), each paired with one of 487 sets of premises which serve as rules to be used to deductively reason for the validity of each conclusion
* 304 stories
* 1,353 NL and FOL premise pairs
* 753 NL and FOL conclusion pairs

### Z-3/3t4 BIG-Bench

* Also called Big Bench Hard
* Diverse evaluation suite that focuses on tasks believed to be beyond the capabilities of current language models
* 23 challenging tasks
* Applying CoT prompting to BBH tasks enables PaLM to surpass the average human rater performance on 10 of the 23 tasks, and Codex (code-davinci-002) to surpass the average human-rater performance on 17 of the 23 tasks
* Task Logical Deduction: Deduce the order of a sequence of objects based on the clues and information about their spatial relationships and placements

### Z-3/3t5 AR-LSAT

* Analytical Reasoning (AR)
* AR-LSAT from the Law School Admission Test (LSAT)
* Solving the problem requires a system to understand the knowledge in the context including participants, positions, rules expressed in natural language (e.g., “If G serves on X, so does B") and facts (e.g., “D and F both serve on the X committee")
* Analytical Reasoning Machine (ARM), a framework that can comprehend the context and perform reasoning for making a conclusion
* 2,046 questions

### Z-3/3t6 T-REx

* Reducing the gap between Natural Language and structured knowledge bases (KB) has been the concern of
* Research tasks such as: Relation Extraction, KB Population, KBdriven Natural Language Generation and QA
* This is a large scale alignment dataset between free text documents and KB triples
* 3.09 million Wikipedia abstracts aligned with 11 million Wikidata triples, covering more than 600 unique Wikidata predicates
* T-REx creation pipeline (Figure 1) contains components for document reading, entity extraction, and dataset exportation into different formats (Document Reader, Entity Extraction, Date and Time Extraction, Predicate Linking, Coreference Resolution, Triple Aligners, Document Writers), while triple aligners – key components of the system – are: NoSub Aligner, AllEnt Aligner, SPO Aligner

### Z-3/3t7 Entailment Bank

* Explain answers by showing the line of reasoning from what is known to the answer
* Generate explanations in the form of entailment trees, namely a tree of multi premise entailment steps from facts that are known, through intermediate conclusions, to the hypothesis of interest (namely the question + answer)
* Contains multistep entailment trees
* Each tree contains an average of 6.6 nodes and 2.7 entailment steps, with the full dataset of 1,840 trees including a range of small and large multi-step entailment problems

### Z-3/3t8 ConceptNet

* ver5… Open Multilingual Graph of General Knowledge
* A knowledge graph that connects words and phrases of natural language with labeled edges
* Knowledge graph version of the Open
* Mind Common Sense project (Singh 2002)
* Designed to represent the general knowledge involved in understanding language, improving natural language applications by allowing the application to better understand the meanings behind the words people use
* Connects words and phrases of natural language (terms) with labeled, weighted edges (assertions)
* 21 million edges
* 8 million node
* English vocabulary contains approximately 1,500,000 nodes, and there are 83 languages in which it contains at least 10,000 nodes

### Z-3/3t9 Microsoft Concept Graph

* KG related work: WordNet, DBpedia, YAGO, Freebase, ConceptNet (Z-3/3t8), NELL, WikiTaxonomy, KnowItAll
* A large taxonomy of terms mined from the internet, with is-a relations between concepts
* End-to-end framework of building and utilizing the Microsoft Concept
Graph, which consists of three major layers, namely semantic network construction, concept conceptualization and applications

### Z-3/3t10 WordNet

* On-line Lexical Database, En
* Words and their meanings, which are organized into synsets (sets of synonyms representing a concept) and linked by semantic relationships
* Organize words into sets of synonyms called "synsets."
* Words are connected by various semantic relationships, such as hypernymy (generalization), hyponymy (specialization), meronymy (part-whole relationships), and antonymy (opposite meanings)
* POS

### Z-3/3t11 GenericsKB

* KB of Generic Statements
* Contains naturally occurring generic sentences, as opposed to extracted or crowdsourced triples
* Culled from over 1.7 billion sentences from three corpora (Waterloo, SimpleWiki, ARC)
* 3.5M statements, each including metadata about its topic, surrounding context, and a confidence measure

### Z-3/3t12 MMLU

* Measuring Massive Multitask Language Understanding
* A massive multitask test consisting of multiple-choice questions from various branches of knowledge
* Spans subjects in the humanities, social sciences, hard sciences, and other areas that are important for some people to learn
* 57 tasks
* Humanities, Social Science, STEM

### Z-3/3t13 MATH

* Mathematics Aptitude Test of Heuristics
* Problems from mathematics competitions including the AMC 10, AMC 12, AIME, and more
* 12,500 problems (7,500 training and 5,000 test)
* Includes AMPS, a diverse pretraining corpus that can enable future models to learn virtually all of K-12 mathematics

### Z-3/3t14 LogiQA

* Machine Reading Comprehension with Logical Reasoning
* Habernal et al.[2018] design a dataset for argument reasoning, where a claim is given and the model is asked to choose a correct premise from two candidates to support the claim
* CLUTRR [2019] is a dataset for inductive reasoning over family relations
* Social relation inference [Bramsen et al., 2011]
* Logical comprehension problems of the National Civil Servant Examination of China, which are designed to test the civil servant candidates’ critical thinking and problem solving
* 13,918 paragraph-question-choice triples with the correct answers
* Categorical reasoning, Sufficient conditional reasoning, Necessary conditional reasoning, Disjunctive reasoning, Conjunctive reasoning
* Methods: Rule-based, DL, Pre-trained, Human

### Z-3/3t15 VQA

* Visual Question Answering
* Task of free-form and open-ended VQA
* Given an image and a NL question about the image, the task is to provide an accurate NL answer
* ~0.25M images
* ~0.76M questions
* ~10M answers
* VQA v2 (a)
* OK-VQA (b)

### Z-3/3t15a VQA v2

* 265,016 images (COCO and abstract scenes)
* At least 3 questions per image
* 10 ground truth answers per question
* 3 plausible (but likely incorrect) answers per question
* Automatic evaluation metric

### Z-3/3t16 OK-VQA

* Where the image content is not sufficient to answer the questions, encouraging methods that rely on external knowledge resources
* 14,000 questions that require external knowledge to answer
* A-OKVQA (a)

### Z-3/3t16a A-OKVQA

* Augmented successor of OK-VQA
* 25K questions requiring a broad base of commonsense and world knowledge to answer

### Z-3/3t17 BLIP-2

* Bootstrapping Language-Image Pre-training (with Frozen Image Encoders and Large Language Models)
* Make use of off-the-shelf image encoders and LLMs for vision language pretraining without affecting these models’ generalization ability
* Bridges the modality gap with a lightweight Querying Transformer, which is pretrained in two stages

### Z-3/3t18 PICa

* Study of GPT-3 for Few-Shot Knowledge-Based VQA
* Prompts GPT3 via the use of Image Captions, for knowledge-based VQA
* Instead of using structured KBs as in previous work, we treat GPT-3 as an implicit and unstructured KB that can jointly acquire and process relevant knowledge
* Convert the image into captions (or tags) that GPT-3 can understand, then adapt GPT-3 to solve the VQA task in a few-shot manner by just providing a few in-context VQA examples

### Z-4 Metrics

* BLeU (a)
* Naturalness (b)
* Likert \(c\)
* GLUE (d)
* Acceptability Judgment (e)
* COPA (f)

### Z-4a BLeU

* BiLingual Evaluation Understudy
* Evaluate the quality of text which has been machine-translated from one natural language to another

### Z-4b Naturalness

* Annotated by native speakers

### Z-4c Likert

* This scale is a unidimensional scale that researchers use to collect respondents' attitudes and opinions

### Z-4d GLUE

* General Language Understanding Evaluation
* Collection of resources for training, evaluating, and analyzing NLU
* Benchmark of nine sentence- or sentence-pair
* Diagnostic

### Z-4e Acceptability Judgment

* Method in empirical linguistics to gather information about the internal grammar of speakers of a language
* Also called: acceptability rating task

### Z-4f COPA

* Choice Of Plausible Alternatives
* One thousand English-language questions that directly assess commonsense causal reasoning
* Each question gives a premise and two plausible causes or effects, where the correct choice is the alternative that is more plausible than the other

### Z-5 Languages

* Nepali (a)
* African (b)
* Hindi \(c\)
* Bengali (d)
* Indonesian (e)

### Z-5a Nepali

* Low-resource
* Devnagari script
* Model: NepBERTa

### Z-5b African

* 1k - 2k languages
* Low-resource

### Z-5c Hindi

* Low-resource
* Devanagari script

### Z-5d Bengali

* Low-resource
* Bengali script

### Z-5d Indonesian

* Low-resource
* Latin (for Bahasa)

### Z-6 Frameworks/Tools

* AllenNLP SRL (a)
* OpenCSR (b)
* M-FleNS \(c\)
* OpenFSP (d)
* Colead (e)
* CLD2 (f)
* Polyglot (g)
* Lingua (h)
* CO3 (i)
* OpenPrompt (j)
* COMBO (k)
* COMET (l)

### Z-6a AllenNLP SRL

* Semantic Role Labeling (SRL) recovers the latent predicate argument structure of a sentence
* “who” did “what” to “whom,”
* Based on: a deep BiLSTM model (He et al, 2017)

### Z-6b OpenCSR

* Open-Ended Common-Sense Reasoning
* Has a dataset and corpus
* Reformatting of QA datasets – ARC, OBQA, and QASC
* Uses GenericsKB

### Z-6c M-FleNS

* An Irish Natural Language Generation system for verbalising part of the DBpedia ontology and building a multilayered dataset with rich linguistic annotations

### Z-6d OpenFSP

* Open Frame Semantic Parser
* Input the developer’s defined functions and their annotations to augment an existing assistant system for new tasks
* Analysing the semantic similarity of slots from TopV2

### Z-6e Colead

* Co-learning Rule and Data from Documentation
* Combine NLU-driven and data-driven synthesizers to exploit their complementary strengths and enable code generation in the presence of data scarcity and domain growth

### Z-6f CLD2

* Compact Language Detector
* Detects over 80 languages in Unicode UTF-8 text
* For mixed-language input, CLD2 returns the top three languages found and their approximate percentages
* Not designed to do well on very short text, at least 200 characters
* Supported: Afrikaans Albanian Arabic Armenian Azerbaijani Basque Belarusian Bengali Bihari Bulgarian Catalan Cebuano Cherokee Croatian Czech Chinese Chinese_T Danish Dhivehi Dutch English Estonian Finnish French Galician Ganda Georgian German Greek Gujarati Haitian_Creole Hebrew Hindi Hmong Hungarian Icelandic Indonesian Inuktitut Irish Italian Javanese Japanese Kannada Khmer Kinyarwanda Korean Laothian Latvian Limbu Lithuanian Macedonian Malay Malayalam Maltese Marathi Nepali Norwegian Oriya Persian Polish Portuguese Punjabi Romanian Russian Scots_Gaelic Serbian Sinhalese Slovak Slovenian Spanish Swahili Swedish Syriac Tagalog Tamil Telugu Thai Turkish Ukrainian Urdu Vietnamese Welsh Yiddish
* Ref: Z-6g

### Z-6g Polyglot

* Google Tool
* Identify more than one single language per document
* Ref: Z-6f

### Z-6h Lingua

* Detect which language some text is written in
* Works with short bits of text
* 75 languages
* These other work with longer bits of text: Google's CLD 2 and CLD 3, Langid, FastText, FastSpell, Simplemma and Langdetect

### Z-6i CO3

* A framework for COntextualizing COmmonsense for distilling COnversations from LLM
* Infuse commonsense knowledge into dialogues by transforming knowledge triples into narratives, and then into dialogues
* LLMs are prone to hallucinations (Weidinger et al., 2021), the seed commonsense knowledge can help them stay on a sensible generation path

### Z-6j OpenPrompt

* Open-source Framework for Prompt-learning
* Directly adapts pre-trained language models (PLMs) to cloze-style prediction, autoregressive modeling, or sequence to sequence generation
* A unified easy-to-use toolkit to conduct prompt-learning over PLMs

### Z-6k COMBO

* Compatibility-Oriented knowledge Merging for Better Open-domain QA framework
* Match LLM-generated passages with retrieved counterparts into compatible pairs, based on discriminators trained with silver compatibility labels
* A Fusion-in-Decoder-based reader model handles passage pairs to arrive at the final answer

### Z-6l COMET

* Crosslingual Optimized Metric for Evaluation of Translation
* Framework for training multilingual machine translation evaluation models

### Z-7 Knowledge

* Triples (a)
* In-context Learning (b)
* Temporal \(c\)
* Information Gain (d)
* Graphs Knowledge (e)
* Relation Extraction (f)

### Z-7a Triples

* Also called “facts”
* Express graph data ie entities and relations between them

### Z-7b In-context Learning

* A specific method of prompt engineering where demonstrations of the task are provided to the model as part of the prompt (in natural language)
* No explicit explicit retraining or fine-tuning needed

### Z-7c Temporal

* Facts, information, or knowledge that are subject to change
* Figuring out what “before” and “after” means

### Z-7d Information Gain

* Paper: On a Measure of the Information Provided by an Experiment, Dennis V. Lindley, 1956
* Measure the effectiveness of a feature in classifying or predicting a target variable

### Z-7e Graphs (Knowledge)

* A graph-structured data model or topology to represent and operate on data

### Z-7f Relation Extraction

* Identification of relations between entities
* Extracting person name, company name, etc

### Z-8 Linguistics

* Code switch/mix (a)
* Code Mixing Index, CMI (b)
* LID \(c\)
* Lexical Ambiguity (d)
* Polysemous Words (e)
* Word Sense (f)
* Multilingual (g)
* Transliteration (h)
* Translation (i)

### Z-8a Code switch/mix

* Alternating between two or more languages or varieties of language in conversation

### Z-8b Code Mixing Index, CMI

* The index can be applied to a sentence and seamlessly extended to a paragraph or an entire document
* Complexity Factor (CF) can be applied to any sentence, paragraph or document which contains multiple languages
* CF = Language Factor (LF), Switching Factor (SF) and Mix Factor (MF)

### Z-8c LID

* Language Identification
* A special case of text categorization, solved with various statistical methods

### Z-8d Lexical Ambiguity

* Multiple interpretations of spoken or written language that renders it difficult or impossible to understand without some additional information
* Contrasted with structural or syntactic ambiguity, which complicates the interpretation of written or spoken language because of the way in which words or phrases are arranged
* Word sense disambiguation (WSD) helps in resolution

### Z-8e Polysemous Words

* Words having more than one meaning
* The occurrence of closely related polysemous words nearby in the word embedding space (i.e. left and right) causes unrelated words to be closer together (e.g. left and wrong)

### Z-8f Word Sense

* A word sense is one of the meanings of a word (“play”)
* “different collocates” can be used to differentiate
* Word-sense disambiguation (WSD) (1)

### Z-8f1 Word-sense disambiguation (WSD)

* The process of identifying which sense of a word is meant in a sentence or other segment of context

### Z-8g Multilingual

Involving multiple languages
* BabelNet (1)

### Z-8g1 BabelNet

* Multilingual lexicalized semantic network and ontology
* Sapienza University of Rome
* Created by linking Wikipedia and WordNet
* Automatic mapping and by filling in lexical gaps in resource-poor languages by using statistical machine translation
* 500 languages

### Z-8h Transliteration 

* A type of conversion of a text from one script to another
* Devnagari to Roman script is called Romanized
* Devnagari to Roman script transliteration can be done losslessly using IAST notation. This is highly standardized

### Z-8i Translation 

* Express the sense of one language into another
* LRP2 (1)

### Z-8i1 LRP2

* Language Representation Projection modules (LRP2)
* The first module converts non-English representations into English-like equivalents, while the second module reverts English-like representations back into representations of the corresponding non-English language

## Workshops Day 6-DEC-2023 WE

### Proceedings of the 2nd Workshop on Pattern-based Approaches to NLP in the Age of Deep Learning

Mihai Surdeanu, Ellen Riloff, Laura Chiticariu, Dayne Frietag, Gus Hahn-Powell, Clayton T. Morrison, Enrique Noriega-Atala, Rebecca Sharp, Marco Valenzuela-Escarcega (Editors)

#### D1-1 [Panel]
* We need to come up with patterns or rules which are machine generated. We cannot audit these patterns
* It must be a symbolic representation, but can we use DL?
* How do we deal with conflicting rules?
* How do we prove the effectiveness of these rules?
* Is a Prompt a rule?
* We need a dataset generator which creates a different distribution as needed “configurable”

#### D1-2 [Keynote] The Role of Patterns in the Era of LLM
Author: Yunyao Li, (Director of Machine Learning, Adobe Experience Platform)
Paper: https://www.slideshare.net/YunyaoLi/the-role-of-patterns-in-the-era-of-large-language-models

* Improve QA - reduce number of unanswered questions
* Introspection based on ontology
* QA-FLEEK
* Factual error detection and correction with evidence retrieved from external knowledge base
* “Direct Triple Retrieval”
* Linking: Offline Entity Linking
* Embedding: Fact Ranking and Related Entities

#### D1-3 Nearest Neighbor Search over Vectorized Lexico-Syntactic Patterns for Relation Extraction from Financial Documents

Author(s): Pawan Rajpoot, Ankur Parikh
Paper: https://aclanthology.org/2023.pandl-1.1.pdf
Code: https://github.com/pawan2411/PAN-DL_Refind
Model: all-mpnet-base-v2
Dataset: REFinD

#### D1-4 LEAF: Linguistically Enhanced Event Temporal Relation Framework
Author(s): Stanley Lim, Da Yin, Nanyun Peng
Paper: https://aclanthology.org/2023.pandl-1.2.pdf
Model(s): LEAF, BERT, RoBERTa
Dataset(s): MATRES, TB-Dense

* “John was cooking freshly made noodles for the family gathering” contains no explicit temporal indicators between the events
* Introduce Linguistically enhanced Event TemporAl relation Framework (LEAF), a simple and effective approach to acquiring rich temporal knowledge of events from large-scale corpora
* “before” and “after” can be used but rarely used. A curated list of patterns can be found in literature and then developed technique to use this data
* Allen NLP SRL parser used
* ChatGPT is very weak with temporal relations
* On MATRES – GPT: 28%, LEAF: 80%
* Headline-generation on a corpus of article pairs: https://huggingface.co/datasets/gigaword

#### D1-5 A Graph-Guided Reasoning Approach for Open-ended Commonsense Question Answering

Author(s): Zhen Han, Yue Feng, Mingming Sun
Paper: https://aclanthology.org/2023.pandl-1.3.pdf
Model(s): DPR, DrKIT, DrFact
Dataset(s): ARC-Open, OBQA-Open

* Benchmark challenge set for open-ended commonsense reasoning (OpenCSR)
* Text2graph using OpenAnnotationTool
* Graph nn for subgraph reasoning
* See OpenCSR

#### D1-6 Generating Irish Text with a Flexible Plug-and-Play Architecture

Author(s): Simon Mille, Elaine Uí Dhonnchadha, Lauren Cassidy, Brian Davis, Stamatia Dasiopoulou, Anya Belz
Paper: https://aclanthology.org/2023.pandl-1.4.pdf
Model(s): gaBERT
Dataset(s): WebNLG

* Specific purpose of building new resources for Irish, a language currently under-represented in the NLP landscape
* Describe M-FleNS, a multi-lingual flexible plug-and-play architecture designed to accommodate neural and symbolic modules
* Pipeline: https://github.com/mille-s/DCU_TCD-FORGe_WebNLG23
* Dataset: https://github.com/mille-s/Mod-D2T/
* Generate Wikipedia pages in Irish or English: https://github.com/mille- s/WikipediaPage_Generator
* M-FleNS = Multilingual Flexible Neuro-Symbolic
* WebNLG for text gen in triples
* See DBpedia triplets

#### D1-7 Symbolic Planning and Code Generation for Grounded Dialogue

Author(s): Justin Chiu, Wenting Zhao, Derek Chen, Saujas Vaduguru, Alexander Rush, Daniel Fried
Paper: https://aclanthology.org/2023.pandl-1.5.pdf
Model(s): SPC, GPT4 2-shot, Imitate
Dataset(s): ONECOMMON

* LLM have had limited applicability in grounded task-oriented dialogue as they are difficult to steer toward task objectives
* Composing LLM with a symbolic planner and grounded code execution
* Consists of a reader and planner: the reader leverages an LLM to convert partner utterances into executable code, calling functions that perform grounding
* Some GPT issues: not multimodal, dealing with uncertainty and ambiguity
* Grounding -> Approach
* Program Synthesis
* OneCommon is a grounding method
* Perspective-dependent ambiguity
* See Expected InfoGain (Lindley, 1956)

#### D1-8 Towards Zero-Shot Frame Semantic Parsing with Task Agnostic Ontologies and Simple Labels

Author(s): Danilo Neves Ribeiro, Jack Goetz, Omid Abdar, Mike Ross, Annie Dong, Kenneth Forbus, Ahmed Mohamed
Paper: https://aclanthology.org/2023.pandl-1.6.pdf
Model(s): RoBERTa
Dataset(s): TopV2

* Frame semantic parsing is a component of task-oriented dialogue systems
* We propose OpenFSP, a framework that allows for easy creation of new domains from a handful of simple labels that can be generated without specific NLP knowledge
* Semantic Parsing Dataset: Topv2
* Adaptation to new domains by non-tech experts

#### D1-9 Co-evolving data-driven and NLU-driven Synthesizers for Generating Code in Domain Growth and Data Scarcity

Author(s): Jiasheng Gu, Zifan Nan, Zhiyuan Peng, Xipeng Shen, Dongkuan Xu
Paper: https://aclanthology.org/2023.pandl-1.7.pdf
Model(s): PyCodeGPT, CodeT5
Dataset(s): Text Editing, Air Travel Information System (ATIS), 

* Data-driven synthesizer requires a large number of query-code pairs for training, which hinders its application to low-resource programming languages
* We propose a circular training framework, Colead, which co-evolves both the data-driven synthesizer and the NLU-driven synthesizer to achieve high-quality code generation in the presence of data scarcity and domain growth

#### D1-10 Complementary Roles of Inference and LM in Q

Author(s): Liang Cheng, Mohammad Javad Hosseini, Mark Steedman
Paper: https://aclanthology.org/2023.pandl-1.8.pdf
Model(s): IE, BERT, GPT-3.5
Dataset(s): LAMA (Google-RE and T-REx)

* The MR-based approach suffers from sparsity issues in extracted knowledge graphs (KGs), while the performance of the LM-based approach significantly depends on the quality of the retrieved context for questions
* A novel methodology that leverages directional predicate entailment (inference) to address these limitations. We use entailment graphs (EGs), with natural language predicates as nodes and entailment as edges, to enhance parsed KGs by inferring unseen assertions, effectively mitigating the sparsity problem in the MR-based approach
* We employ a coreference resolution tool (Lee et al., 2018) to handle coreferences of texts, and then follow Hosseini et al. (2018) and use GraphParser (Reddy et al., 2014) to extract triples from the processed text. GraphParser utilizes a combinatory categorial grammar (CCG) parser (Steedman, 2000) to convert sentences into semantic graphs, which are subsequently transformed into triples. GraphParser: https://github.com/sivareddyg/graph-parser, KG construction: https://github.com/LeonChengg/entGraphQA.git

#### D1-11 Controlled Data Augmentation for Training Task-Oriented Dialog Systems with Low Resource Data

Author(s): Sebastian Steindl, Ulrich Schäfer, Bernd Ludwig
Paper: https://aclanthology.org/2023.pandl-1.9.pdf
Model(s): BART Base
Dataset(s): MultiWOZ 2.4

* DL rely on large amounts of training data
* Collection of conversational data is often a tedious and costly process
* Our method generates utterances based on dialog annotations in a sequence-to-sequence manner

#### D1-12 A Hybrid of Rule-based and Transformer-based Approaches for Relation Extraction in Biodiversity Literature

Author(s): Roselyn Gabud, Portia Lapitan, Vladimir Mariano, Eduardo Mendoza, Nelson Pampolina, Maria Art Antonette Clariño, Riza Batista-Navarro
Paper: https://aclanthology.org/2023.pandl-1.10.pdf
Model(s): T5
Dataset(s): Derived from work of Gabud et al. (2019) and was designed in accordance with the annotation scheme used in the COPIOUS project (Nguyen et al., 2019).

* Relation extraction (RE) is one of the tasks behind many relevant natural language processing (NLP) applications
* Advantage of the zero-shot (i.e., not requiring any labeled data) capability of pattern-based methods for RE using a rule-based approach, combined with templates for natural language inference (NLI) transformer models
* We present our hybrid method for RE that exploits the advantages of both methods, i.e., interpretability of rules and transferability of transformers

## Tutorials Day 7-DEC-2023 TH

### Proceedings of the 6th Workshop on Computational Approaches to Linguistic Code-Switching

Genta Winata, Sudipta Kar, Marina Zhukova, Thamar Solorio, Mona Diab, Sunayana Sitaram, Monojit Choudhury, Kalika Bali

#### D2-1 TongueSwitcher: Fine-Grained Identification of German-English Code-Switching

Author(s): Igor Sterner, Simone Teufel
Paper: https://aclanthology.org/2023.calcs-1.1.pdf
Code: https://huggingface.co/igorsterner/german-english-code-switching-bert
Model(s): bert-base-multilingual-cased
Dataset(s): TONGUESWITCHER

* We provide the largest corpus of naturally occurring German-English code-switching
* The first method is rule-based, using wordlists and morphological processing
* In our second method, we continue pretraining of a neural LM on this corpus and classify tokens based on embeddings from this LM
* Use of tweet based dataset
* Polyglot is another such tool, which is able to identify more than one single language per document (Chen and Skiena, 2014). It is built from the CLD2 tool from Riesa and Giuliani (2013), which uses quadgram ranking
* Lingua (Stahl, 2023) is a black-box LI tool that also offers code-switching identification for many language pairs, including German–English
* If the same string “was” appeared in English, it would be the past form of ‘to be’. Importantly, the two meanings are entirely unrelated. Such cases constitute an interesting corner case for code-switched text, and are called interlingual homographs (IHs, Dijkstra et al., 1999)
* Different patterns of CS. Patterns of CS may change over generations
* Nguyen et al. (2020; 2021) present rule-based code-switching identification systems for Vietnamese–English and Hindi–English mixed text, which is based on specially-created wordlists for each of these language pairs

#### D2-2 Towards Real-World Streaming Speech Translation for Code-Switched Speech

Author(s): Belen Alastruey, Matthias Sperber, Christian Gollan, Dominic Telaar, Tim Ng, Aashish Agarwal
Paper: https://aclanthology.org/2023.calcs-1.2.pdf
Code: https://github.com/apple/ml-codeswitching-translations
Model(s): Multimodal model design proposed by Ye et al. (2021), wav2vec2-base-960h
Dataset(s): Bangor Miami, Fisher, CoVoST, MuST-C

* We focus on two essential yet unexplored areas for real-world CS speech translation: streaming settings, and translation to a third language (i.e., a language not included in the source)
* See datasets mentioned in paper
* Use of some synthetic data
* Updations are made in realtime
* Predictive model
* Acc vs Latency
* CS has impact in translations
* Can BLEU be used as a metric?

#### D2-3 Language Preference for Expression of Sentiment for Nepali-English Bilingual Speakers on Social Media

Author(s): Niraj Pahari, Kazutaka Shimada
Paper: https://aclanthology.org/2023.calcs-1.3.pdf
Model(s): mBERT, XLM-R, MuRIL

* We aim to study the language preference of multilingual Nepali-English CS speakers while expressing sentiment in social media
* We create a novel dataset for sentiment analysis using the public Nepali-English code-switched comments in YouTube
* Emotion and swearing
* Code Mixing Index (CMI) (Das and Gambäck, 2014)

#### D2-4 Text-Derived Language Identity Incorporation for End-to-End Code-Switching Speech Recognition

Author(s): Qinyi Wang, Haizhou Li
Paper: https://aclanthology.org/2023.calcs-1.4.pdf
Model(s): Language identity-language model, 
Dataset(s): SEAME

* Language identity (LID) is often integrated into the speech recognition system to provide additional linguistic context
* We introduce a novel approach to learn language identity from pure text data via a dedicated language identity-LM
* This paper has a good and simple overview of the problem
* Transformer-based architecture
* Ref paper for model architecture in particular how “gating” is used to derive info

#### D2-5 Code-Switching with Word Senses for Pretraining in Neural Machine Translation

Author(s): Vivek Iyer, Edoardo Barba, Alexandra Birch, Jeff Pan, Roberto Navigli
Paper: https://aclanthology.org/2023.findings-emnlp.859.pdf

* Lexical ambiguity is a significant and pervasive challenge in NMT
* NMT systems struggling to handle polysemous words (Campolungo et al., 2022)
* We introduce Word Sense Pretraining for Neural Machine Translation (WSP-NMT) - an end-to-end approach for pretraining multilingual NMT models leveraging word sense-specific information from Knowledge Bases
* “Code Switched Pretraining”
* “Sense-pivoted pretraining” at sentence level, not word level
* BabelNet (Navigli et al., 2021) is in lemmatized mode
* See: WSD (Word Sense Disambiguation) system, ESCHER (Barba et al., 2021a)

#### D2-6 Prompting Multilingual LLM to Generate Code-Mixed Texts: The Case of South East Asian Languages

Author(s): Zheng Xin Yong, Ruochen Zhang, Jessica Forde, Skyler Wang, Arjun Subramonian, Holy Lovenia, Samuel Cahyawijaya, Genta Winata, Lintang Sutawika, Jan Christian Blaise Cruz, Yin Lin Tan, Long Phan, Long Phan, Rowena Garcia, Thamar Solorio, Alham Aji
Paper: https://aclanthology.org/2023.calcs-1.5.pdf
Model(s): ChatGPT, InstructGPT, BLOOMZ, Flan-T5-XXL
Dataset(s): SEA

* The differences in decision making between behavioural models of voice interfaces are hard to capture using existing measures for the absolute performance of such models
* We propose a general methodology to compute the similarity of two dialogue behaviour models and investigate different ways of computing scores on both the semantic and the textual level
* Can we generate text in CS mode?
* We collect synthetic code-mixed data by prompting LLM with requests along two axes: languages (7) and topics (food, family, traffic, Artificial Intelligence, and weather)
* 6 prompt templates
* Measurements: Level of CS, Naturalness, Accurateness, 

#### D2-7 CONFLATOR: Incorporating Switching Point based Rotatory Positional Encodings for Code-Mixed LM

Author(s): Mohsin Mohammed, Sai Kandukuri, Neeharika Gupta, Parth Patwa, Anubhab Chatterjee, Vinija Jain, Aman Chadha, Amitava Das
Paper: https://aclanthology.org/2023.calcs-1.6.pdf
Model(s): CONFLATOR
Dataset(s): ICON

* How much data do we have? Performance depends on this
* Ref paper: PESTO: Switching Point based Dynamic and Relative Positional Encoding for Code-Mixed Languages, https://arxiv.org/pdf/2111.06599.pdf
* Rotate with text and invert on reaching switching point index
* See heatmap for visualization of switching point. We can also use similar visualizations for other purposes
* We hypothesize that Switching Points (SPs), i.e., junctions in the text where the language switches (L1 -> L2 or L2 -> L1), pose a challenge for CM LM, and hence give special emphasis to SPs in the modeling process
* We experiment with several positional encoding mechanisms and show that rotatory positional encodings along with switching point information yield the best results
* We introduce CONFLATOR: a neural LM approach for code-mixed languages
CONFLATOR tries to learn to emphasize switching points using smarter positional encoding, both at unigram and bigram levels

#### D2-8 Unified Model for Code-Switching Speech Recognition and Language Identification Based on Concatenated Tokenizer

Author(s): Kunal Dhawan, KDimating Rekesh, Boris Ginsburg
Paper: https://aclanthology.org/2023.calcs-1.7.pdf
Code: https://github.com/NVIDIA/NeMo/tree/main/scripts/speech_recognition/code_switching
Model(s): Conformer-RNNT Large, 
Dataset(s): LibriSpeech, Multilingual LibriSpeech, Voxpopuli, Fisher, ULCA, Miami Bangor, MUCS

* Propose (1) a new method for creating code-switching ASR datasets from purely monolingual data sources, and (2) a novel Concatenated Tokenizer that enables ASR models to generate language ID for each emitted text token while reusing existing monolingual tokenizers
* FLEURS: Few-shot learning evaluation of universal representations of speech. FLEURS dataset used
* Reuse model tokenizer

#### D2-9 [Panel]

* Need of a multilingual dataset
* Research in CS in GenAI with HCI
* Type of communication: Convo / Social network / Text
* GPT might be “accidentally” multilingual. We don’t want this CS if it happens
* “Naturalness” vs “Acceptability Judgement”

#### D2-10 Multilingual self-supervised speech representations improve the speech recognition of low-resource African languages with codeswitching

Author(s): Tolulope Ogunremi, Christopher Manning, Dan Jurafsky
Paper: https://aclanthology.org/2023.calcs-1.8.pdf
Dataset(s): Code-switched soap opera speech, Kaldi-based

* Can large multilingual models be used for translation to nearby languages
* SA soap opera scripts being used for CS
* Tags and casing used to encode language info
* We propose fine-tuning self-supervised speech representations such as wav2vec 2.0 XLSR to recognize code-switched data

#### D2-11 [Keynote] Resource-efficient Computational Models for Code-switched Speech and Text

Author(s): Preethi Jyothi, IIT Bombay

* ChatGPT is not competitive for minority MT tasks
* The problem is CS and transliteration
* Can we not treat this a new language
* Use of “synthesis” prior to pre-training to create bilingual data for training. This should be realistic text using bilingual or parallel datasets
* Use noise to do denoising (MLM type)
* CS text
* Hi-En dataset
* Movie CS, Treebank CS
* Syntactic / Semantic / Naturalness – as a measure of dataset
* Also measured on the Likert scale, BLEU / LM perplexity, GLUE COS / BERT score / BERT classifier
* Diversity in CS: Socio-linguistics (1st gen immigrants vs younger immigrants)
* Formality in rendered text
* Recall diversity
* How to measure how much CS has occurred
* MLM pretraining
* “Frequency Effect” normalizes / desensitizes
* Ref: CMI

#### D2-12 Modeling Code-Switch Languages Using Bilingual Parallel Corpus

Author(s): Grandee Lee, Haizhou Li
Paper: https://aclanthology.org/2020.acl-main.80.pdf
Model(s): BALM (attention-based, auto- regressive model, bilingual attention language model)
Dataset(s): SEAME

* See SCDF example
* Phonetic and Phonotactic difference of languages. Ref: A Phonotactic LM for Spoken Language Identification, by Haizhou Li and Bin Ma, https://dl.acm.org/doi/pdf/10.3115/1219840.1219904
* The “Matrix Language Frame theory” (Myers-Scotton, 1997), which shows that individual monolingual sentences will conform to the grammar of the matrix language
* The “Equivalence Constraint theory” (Poplack, 2000; Sankoff, 1998), which further constrains the intra-sentential CS points to the syntactic boundaries shared by both languages
* The “Functional Head Constraint theory” (Di Sciullo et al., 1986; Belazi et al., 1994) that imposes constraints on the functional head and its complements
* Data prep like so: En, Hi, Eng+Hi, Hi+Eng

## Day 1 Main Conf 8-DEC-2023 FR

#### D3-1 Learning Retrieval Augmentation for Personalized Dialogue Generation

Author(s): Qiushi Huang, Shuai Fu, Xubo Liu, Wenwu Wang, Tom Ko, Yu Zhang, Lilian Tang
Paper: https://aclanthology.org/2023.emnlp-main.154.pdf
Code: https://github.com/hqsiswiliam/LAPDOG
Model(s): LAPDOG
Dataset(s): CONVAI2, ROCStories, PersonaChat




* Persona is a few sentences. Persona may be too little data
* We need a “story” and how do we learn these Retrieval Augmented stories?
* Create a “retriever” which takes persona and combines with a Stories Corpus which gives “stories”
* Ref: Architecture Diagram
* Partial use of random stories to avoid fixed sets
* Ref:  Fusion-in-Decoder (FiD) technique (Izacard and Grave, 2021) to integrate the retrieved contents with the persona and dialogue history

#### D3-2 Prompt-Based Monte-Carlo Tree Search for Goal-oriented Dialogue Policy Planning
Author(s): Xiao Yu, Maximillian Chen, Zhou Yu
Paper: https://arxiv.org/pdf/2305.13660.pdf
Code: https://github.com/jasonyux/GDPZero
Model(s): ChatGPT
Dataset(s): PersuasionForGood

* Policy = next best action
* Human strategies have a lot of variance
* Difficult to train and collect
* Not “what to plan” or “how to plan”
* Propose: move > do > eval
* MCTS is used for “think ahead”
* [1] MCTS with zero training; LLM can be a value for
* [2] Open-loop MCTS for dialog (deal with stochasticity) GCP-Zero
* Evaluate quality of response (not just utterances)
* Ref: Architecture
* Pure play LLM usage in the PersuasionForGood dataset

#### D3-3 MADNet: Maximizing Addressee Deduction Expectation for Multi-Party Conversation Generation

Author(s): Jia-Chen Gu, Chao-Hong Tan, Caiyuan Chu, Zhen-Hua Ling, Chongyang Tao, Quan Liu, Cong Liu
Paper: https://arxiv.org/pdf/2305.12733.pdf
Model(s): MADNet, 
Dataset(s): Ubuntu IRC 

* A two-party convo is sequential, a multi-party convo is graphical flow
* “conversation graph” is used for representation
* MPC = multi-party conversation
* Ref: Training process

#### D3-4 SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization

Author(s): Hyunwoo Kim, Jack Hessel, Liwei Jiang, Peter West, Ximing Lu, Youngjae Yu, Pei Zhou, Ronan Le Bras, Malihe Alikhani, Gunhee Kim, Maarten Sap, Yejin Choi
Paper: https://aclanthology.org/2023.emnlp-main.799.pdf
Code: https://github.com/skywalker023/sodaverse
Dataset(s): SODA

* SODA = SOcial DiAlogues, a million-scale English dialogue dataset covering a wide variety of social interactions
* CO3: A Contextualization Framework for Conversation Distillation using Commonsense

#### D3-5 What Comes Next? Evaluating Uncertainty in Neural Text Generators Against Human Production Variability

Author(s): Mario Giulianelli, Joris Baan, Wilker Aziz, Raquel Fernández, Barbara Plank
Paper: https://arxiv.org/pdf/2305.11707.pdf
Code: https://github.com/dmg-illc/nlg-uncertainty-probes
Model(s): Helsinki-NLP TransformerAlign, T5, GPT2-large, DialoGPT-medium, 
Dataset(s): WMT-14 En-De, ASSET, WritingPrompts, DailyDialog++, 

* Characterize human production: lexically, syntactically, and semantically

#### D3-6 Enhancing Code-Switching for Cross-lingual SLU: A Unified View of Semantic and Grammatical Coherence

Author(s): Zhihong Zhu, Xuxin Cheng, Zhiqi Huang, Dongsheng Chen, Yuexian Zou
Paper: https://aclanthology.org/2023.emnlp-main.486.pdf
Model(s): SoGo, mBERT, ZSJoint, SoSDA, GL-CLEF, LAJ-MCL, CoSDA_XLM-based, CoSDA_mBERT-based
Dataset(s): MultiATIS++

* Characterize human production: lexically, syntactically, and semantically
* SOGO = semantics-coherent and grammar-coherent method
* SoGo to enhance code-switching for zero-shot cross-lingual SLU
* Ref: Idea
* Perform token-level alignment in lower six layers of mBERT to enhance the grammatical coherence of the code-switched sentence

#### D3-7 Learning to Predict Task Transferability via Soft Prompt

Author(s): Lingyun Feng
Paper: https://aclanthology.org/2023.emnlp-main.546.pdf
Model(s): RoBERTa-large
Dataset(s): Cosmos QA, SWAG, DuoRC, etc. (48 tasks)

* To learn an affinity scoring function to predict transferability between tasks

#### D3-8 Crystal: Introspective Reasoners Reinforced with Self-Feedback

Author(s): Jiacheng Liu, Ramakanth Pasunuru, Hannaneh Hajishirzi, Yejin Choi, Asli Celikyilmaz
Paper: https://arxiv.org/pdf/2310.04921.pdf
Code: https://github.com/liujch1998/crystal
Model(s): https://huggingface.co/liujch1998/crystal-11b
Dataset(s): OpenBookQA, ARC, CommonsenseQA, etc. (10 total)

* Performance and interpretability of commonsense reasoning can be improved via knowledge-augmented reasoning methods
* CoT is not enough
* Introspect for knowledge statements related to the given question, then make an informed prediction that is grounded in the previously introspected knowledge
* K​nowledge introspection and knowledge-grounded reasoning modes of the model are tuned via reinforcement learning
* Demo: https://huggingface.co/spaces/liujch1998/crystal

#### D3-9 Improving Dialogue Discourse Parsing via Reply-to Structures of Addressee Recognition

Author(s): Yaxin FAN, Feng Jiang, PEIFENG LI, Fang Kong, Qiaoming Zhu
Paper: https://aclanthology.org/2023.emnlp-main.526.pdf
Code: https://github.com/yxfanSuda/RLTST
Model(s): ChatGPT, DSM, SSAM, etc.
Dataset(s): STAC, Molweni

* Jointly learn dialogue discourse parsing with related tasks
* Integrate dialogue discourse parsing with neighboring task addressee recognition. Addressee recognition reveals the reply-to structure that partially overlaps with the relation-based structure, which can be exploited to facilitate relation-based structure learning

#### D3-10 DialCoT Meets PPO: Decomposing and Exploring Reasoning Paths in Smaller LM

Author(s): Chengcheng Han, Xiaowei Du, Che Zhang, Yixin Lian, Xiang Li, Ming Gao, Baoyuan Wang
Paper: https://aclanthology.org/2023.emnlp-main.501.pdf
Code: https://github.com/hccngu/DialCoT
Model(s): FlanT5-XL, code-davinci-002, LaMDA137B, PaLM-60B, UL2-20B
Dataset(s): GSM8K, MultiArith, ASDiv,  SVAMP

* Model’s reasoning path selection uses PPO

#### D3-11 The Framework Tax: Disparities Between Inference Efficiency in NLP Research and Deployment

Author(s): Jared Fernandez, Jacob Kahn, Clara Na, Yonatan Bisk, Emma Strubell
Paper: https://aclanthology.org/2023.emnlp-main.98.pdf
Code: https://github.com/JaredFern/Framework-Tax

* Improvement in computational throughput and reductions in floating point operations have not directly translated to improvements in wall-clock inference latency
* This is due to deep learning frameworks
* For production, recommendations to researchers and practitioners aimed at narrowing the gap

#### D3-12 Just Adjust One Prompt: Enhancing In-Context Dialogue Scoring via Constructing the Optimal Subgraph of Demonstrations and Prompts

Author(s): Jiashu Pu, ling Cheng, Lu Fan, Tangjie Lv, Rongsheng Zhang
Paper: https://aclanthology.org/2023.emnlp-main.590.pdf
Code: https://github.com/iamlxb3/EMNLP2023-ADOROR
Model(s): gpt-3.5-turbo, GPT-4, 
Dataset(s): DailyDialog-Zhao, Persona-Zhao, Topical-USR, Persona-USR, FED (Dialogue)

* Dimension-agnostic scoring method that leverages the in-context learning (ICL) capability of LLM
* [1] Automatically generating prompts, allowing the LLM to observe human labels and summarize the most suitable prompt
* [2] LLM has a token limit and ICL is sensitive to demonstration variations, train a selector to finely customize demonstrations and prompts for each dialogue input
* [3] During inference, request the LLM multiple times with a subgraph of demonstrations and prompts that are diverse and suitable to maximize ICL from various human scoring

#### D3-13 An Integrative Survey on Mental Health Conversational Agents to Bridge Computer Science and Medical Perspectives

Author(s): Young Min Cho, Sunny Rai, Lyle Ungar, João Sedoc, Sharath Chandra Guntuku
Paper: https://aclanthology.org/2023.emnlp-main.698.pdf
Code: https://github.com/JeffreyCh0/mental_chatbot_survey

* See the git repo for details

#### D3-14 From Multilingual Complexity to Emotional Clarity: Leveraging Commonsense to Unveil Emotions in Code-Mixed Dialogues

Author(s): Shivani Kumar, Ramaneswaran S, Md Shad Akhtar, Tanmoy Chakraborty
Paper: https://aclanthology.org/2023.emnlp-main.598.pdf
Code: https://github.com/LCS2-IIITD/EMNLP-COFFEE
Model(s): BERT, RoBERTa, mBERT, MURIL, CoMPM, DialogXL
Dataset(s): https://github.com/LCS2-IIITD/EMNLP-COFFEE/tree/main/Data

* Understand emotions during conversation
* Pipeline to extract commonsense from existing knowledge graphs based on code-mixed

#### D3-15 e-THERAPIST: I suggest you to cultivate a mindset of positivity and nurture uplifting thoughts

Author(s): Kshitij Mishra, Priyanshu Priya, Manisha Burja, Asif Ekbal
Paper: https://aclanthology.org/2023.emnlp-main.861.pdf
Code: https://github.com/Mishrakshitij/e-THERAPIST.git
Model(s): GPT-2 medium
Dataset(s): PSYCON

* Must exhibit politeness and empathy
* Vary as per the user’s gender, age, persona, and sentiment
* A novel polite interpersonal psychotherapy dialogue system to address issues like depression, anxiety, schizophrenia
* Ref: Architecture Diagram

#### D3-16 ReSee: Responding through Seeing Fine-grained Visual Knowledge in Open-domain Dialogue

Author(s): Haoqin Tu, Yitong Li, Fei Mi, Zhongliang Yang
Paper: https://aclanthology.org/2023.emnlp-main.479.pdf
Code: https://github.com/ImKeTT/ReSee
Model(s): RESEE
Dataset(s): WoW, DD

* Adding visual knowledge into text-only dialogue systems has become a potential direction to imitate the way humans think, imagine, and communicate
* Framework RESEE to add visual representation into vanilla dialogue models by modality concatenations

#### D3-17 PK-ICR: Persona-Knowledge Interactive Multi-Context Retrieval for Grounded Dialogue

Author(s): Minsik Oh, Joosung Lee, Jiwei Li, Guoyin Wang
Paper: https://aclanthology.org/2023.emnlp-main.1020.pdf
Code: https://github.com/minsik-ai/PK-ICR
Model(s): Sentence-BERT, DistillBERT, TAS-B
Dataset(s): Call For Customized Conversation, MiniLM, MS MARCO

* Identifying relevant persona or knowledge for conversational systems is critical to grounded dialogue response generation
* [1] Persona and knowledge dual context retrieval methodology
* [2] Framework for cross-task adaptation of dialogue context interactions
* [3] Evaluating the hard-negative trait of Persona-augmented Dialogue

## Day 2 Main Conf 9-DEC-2023 SA

#### D4-1 APrompt: Attention Prompt Tuning for Efficient Adaptation of Pre-trained LM

Author(s): Qifan Wang, Yuning Mao, Jingang Wang, Hanchao Yu, Shaoliang Nie, Sinong Wang, Fuli Feng, Lifu Huang, Xiaojun Quan, Zenglin Xu, Dongfang Liu
Paper: https://aclanthology.org/2023.emnlp-main.567.pdf
Model(s): T5
Dataset(s): BoolQ, CB, COPA, MRC, ReC, RTE, WiC, WSC

* Incorporates query, key, and value prompts into the attention layer to guide the attention computation during fine-tuning
* APROMPT is implemented with the OpenPrompt

#### D4-2 Reasoning with LM is Planning with World Model

Author(s): Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, Zhiting Hu
Paper: https://aclanthology.org/2023.emnlp-main.507.pdf
Code: https://github.com/Ber666/llm-reasoners
Model(s): LLaMA-33B
Dataset(s): Blocksworld, ​​GSM8k

* Incorporates query, key, and value prompts into the attention layer to guide the attention computation during fine-tuning
* LLM reasoning framework, Reasoning via Planning (RAP). RAP repurposes the LLM as both a world model and a reasoning agent, and incorporates a principled planning algorithm (based on Monte Carlo Tree Search) for strategic exploration in the vast reasoning space
* Ref: https://www.llm-reasoners.net/

#### D4-3 Automatic Transcription of Handwritten Old Occitan Language

Author(s): Esteban Garces Arias, Vallari Pai, Matthias Schöffel, Christian Heumann, Matthias Aßenmacher
Paper: https://aclanthology.org/2023.emnlp-main.953.pdf
Code: https://github.com/EstebanGarces/OcciGen
Model(s): BEiT, DeiT, ViT, Swin, GPT-2, BERT
Dataset(s): MLW_data, dom_project

* Handwritten Text Recognition (HTR) for high-resource languages and standardized/machine-written text, their application to low-resource languages often presents challenges
* The model combines a custom-trained Swin image encoder with a BERT text decoder, which we pre-train using a large-scale augmented synthetic data set and fine-tune on the small human-labeled data set
* Misoda Working Group: https://huggingface.co/misoda

#### D4-4 Don’t Trust ChatGPT when your Question is not in English: A Study of Multilingual Abilities and Types of LLM

Author(s): Xiang Zhang, Senyu Li, Bradley Hauer, Ning Shi, Grzegorz Kondrak
Paper: https://aclanthology.org/2023.emnlp-main.491.pdf
Model(s): ChatGPT
Dataset(s): GSM8K, CommonsenseQA, WebQuestions, JOKER@CLEF 2022

* Performance varies across different languages
* Propose a systematic way of qualitatively and quantitatively evaluating the multilingual capabilities of LLM
* Employ a novel prompt back-translation method
* Datasets at: https://github.com/Senyu-Li/LLM-Multilingual-Types

#### D4-5 Revisiting Machine Translation for Cross-lingual Classification

Author(s): Mikel Artetxe, Vedanuj Goswami, Shruti Bhosale, Angela Fan, Luke Zettlemoyer
Paper: https://aclanthology.org/2023.emnlp-main.399.pdf
Model(s): RoBERTa,  XLM-R, DeBERTaV3 large, NLLB
Dataset(s): XNLI, PAWS-X, MARC, XCOPA, XStoryCloze, EXAMS

* [1] zero-shot transfer
* [2] translate-train is an extension of this method that augments the downstream training data by translating it to all target languages through MT
* [3] translate-test, uses MT to translate the test data into English, and runs inference
using an English-only model

#### D4-6 Language Representation Projection: Can We Transfer Factual Knowledge across Languages in Multilingual LM?

Author(s): Shaoyang Xu, Junzhuo Li, Deyi Xiong
Paper: https://aclanthology.org/2023.emnlp-main.226.pdf
Model(s): mBERT, mBERT (LRP2), BLOOM, BLOOM (LRP2)
Dataset(s): mLAMA, TREx, OPUS-100

* Substantial performance gap of factual knowledge probing exists between high-resource languages and low-resource languages
* Propose two parameter-free Language Representation Projection modules (LRP2)
* OPUS (open parallel corpus): https://opus.nlpl.eu/

#### D4-7 Multilingual LLM Are Not (Yet) Code-Switchers

Author(s): Ruochen Zhang, Samuel Cahyawijaya, Jan Christian Blaise Cruz, Genta Indra Winata, Alham Fikri Aji
Paper: https://aclanthology.org/2023.emnlp-main.774.pdf
Model(s): BLOOMZ, mT0, XGLM, ChatGPT, XLM-RoBERTa, mBERT, mDeBERTa v3, M2M100, mBART-50
Dataset(s): MixMT 2022 shared task, Gupshup, SAMSum, LinCE benchmark, Sentimix Spanish-English, MixSentiment Malayalam, MixSentiment Tamil

* Substantial performance gap of factual knowledge probing exists between high-resource languages and low-resource languages
* Multilingual LLM exhibiting promising outcomes in certain tasks using zero or few-shot prompting, they still underperform in comparison to fine-tuned models of much smaller scales

#### D4-8 Merging Generated and Retrieved Knowledge for Open-Domain QA

Author(s): Yunxiang Zhang, Muhammad Khalifa, Lajanugen Logeswaran, Moontae Lee, Honglak Lee, Lu Wang
Paper: https://aclanthology.org/2023.emnlp-main.286.pdf
Code: https://github.com/yunx-z/COMBO
Model(s): InstructGPT, ChatGPT, RoBERTa-large, DeBERTa-large
Dataset(s): NaturalQuestions, TriviaQA, WebQuestion, HotpotQA


* Retrieving passages suffer from insufficient knowledge coverage
* Prompting LLM to generate contextual passages based on their parametric knowledge improves QA performance
* LLM tend to “hallucinate” content that conflicts with the retrieved knowledge
Propose COMBO, a Compatibility-Oriented knowledge Merging for Better Open-domain QA framework (merge LLM and KB)
* A Fusion-in-Decoder (Izacard and Grave, 2021b) based reader model handles passage pairs to arrive at the final answer common retrieve-then-read framework (Izacard and Grave, 2021a,b; Izacard et al., 2022; Karpukhin et al., 2020)

#### D4-9 BanglaAbuseMeme: A Dataset for Bengali Abusive Meme Classification

Author(s): Mithun Das, Animesh Mukherjee
Paper: https://aclanthology.org/2023.emnlp-main.959.pdf
Code: https://github.com/hate-alert/BanglaAbuseMeme
Model(s): BERT, VGG16, MLP, m-BERT, MuRIL, BanglaBERT, XLM-R, CLIP(L)
Dataset(s): BanglaAbuse-Meme

* Easy way to abuse individuals or communities is by creating memes
* Threat to online safety
* Challenging in a low-resource setting
* Data labeling tool: Label Studio
* Exposure to online abuse could usher in unhealthy mental health issues. The annotators were advised to take frequent breaks and not do the annotations in one sitting. Also weekly meetings with them to ensure that the annotations did not affect their mental health

#### D4-10 The Art of SOCRATIC QUESTIONING: Recursive Thinking with LLM

Author(s): Jingyuan Qi, Zhiyang Xu, Ying Shen, Minqian Liu, Di Jin, Qifan Wang, Lifu Huang
Paper: https://aclanthology.org/2023.emnlp-main.255.pdf
Code: https://github.com/VT-NLP/SOCRATIC-QUESTIONING
Model(s): GPT-3, ChatGPT
Dataset(s): Massive Multitask Language Understanding -Physics and Chemistry  (MMLU), MATH (DA), LogiQA, VQA-V2, OK-VQA, AOK- VQA, BLIP- 2, PICa

* CoT is confined by single-pass and sequential generation process, relies on the initial decisions, errors in early steps accumulate and impact the final answers
* Humans adopt recursive thinking i.e. iteratively breaking the original problem into approachable sub-problems and aggregating their answers to resolve the original one
* SOCRATIC QUESTIONING (SQ), a divide-and-conquer style algorithm that mimics the recursive thinking process

#### D4-11 On the Robustness of Dialogue History Representation in Conversational Question Answering: A Comprehensive Study and a New Prompt-based Method

Author(s): Roi Reichart, Zorik Gekhman, Nadav Oved, Orgad Keller, Idan Szpektor
Paper: https://arxiv.org/pdf/2206.14796.pdf
Code: https://github.com/zorikg/MarCQAp
Model(s): Longformer, CONCAT, REWRITE, REWRITEC, ExCorDLF, HAELF, PosHAELF, 
Dataset(s): QuAC, CoQA, DoQA, QuAC Noisy-History (QuAC-NH)

* We design MarCQAp, a novel prompt-based history modeling approach that highlights answers from previous conversation turns by inserting textual prompts in their respective positions within P
* Prompting often refers to the practice of adding phrases to the input, in order to encourage pre-trained LMs to perform specific tasks
* MarCQAp closely resembles the prompting approach from Ben-David et al. (2022) since our prompts are: (1) discrete (i.e the prompt is an actual text-string), (2) dynamic (i.e example-based), and (3) added to the input text and the model then makes predictions conditioned on the modified input

## Day 3 Main Conf 10-DEC-2023 SU

#### D5-1 Explicit Planning Helps LM in Logical Reasoning

Author(s): Hongyu Zhao, Kangrui Wang, Mo Yu, Hongyuan Mei
Paper: https://aclanthology.org/2023.emnlp-main.688.pdf
Code: https://github.com/cindermond/leap
Model(s): T5, GPT-3.5
Dataset(s): PrOntoQA, Entailment Bank, QASC

* Propose LEAP, a novel system that uses LM to perform multi-step logical reasoning and incorporates explicit planning into the inference procedure
* Propose a training strategy that safeguards the planning process from being led astray by spurious features
* Model-based reinforcement learning uses environment models to simulate responses to actions and then uses the simulated experiences to help learn value functions

#### D5-2 Where to start? Analyzing the potential value of intermediate models

Author(s): Leshem Choshen, Elad Venezian, Shachar Don-Yehiya, Noam Slonim, Yoav Katz
Paper: https://aclanthology.org/2023.emnlp-main.90.pdf
Code: https://ibm.github.io/model-recycling/
Model(s): RoBERTa-base, T5-small
Dataset(s): General: {GLUE: CoLA, SST2, MRPC, QQP, MNLI, QNLI, RTE, WNLI, SuperGLUE: BoolQ, CB, CoPA, MULTIRC, WIC, WSC}, NLI: {MNLI, QNLI, RTE, WNLI, ESNLI, adversarial NLI}, Twitter: {EmoInt, Emoji, Irony, OffenseEval, HatEval, Sentiment Analysis}

* Perform a systematic analysis of this “intertraining scheme”

#### D5-3 What do Deck Chairs and Sun Hats Have in Common? Uncovering Shared Properties in Large Concept Vocabularies

Author(s): Amit Gajbhiye, Zied Bouraoui, Na Li, Usashi Chatterjee, Luis Espinosa-Anke, Steven Schockaert
Paper: https://aclanthology.org/2023.emnlp-main.654.pdf
Code: https://github.com/amitgajbhiye/concept_commonality
Model(s): BERT, RoBERTa, DeBERTa
Dataset(s): McRae, CSLB

* Propose a strategy for identifying what different concepts, from a potentially large concept vocabulary, have in common with others
* The only large-scale knowledge base that contains such training examples is ConceptNet, which is unfortunately rather noisy and imbalanced
* A large set of (hyponym and hypernym) pairs from Microsoft Concept Graph, with examples from GenericsKB
* Converted instances of the relations IsA, PartOf, LocatedAt, UsedFor and HasProperty into a set of 63,872 (concept,property) pairs

#### D5-4 AdaSent: Efficient Domain-Adapted Sentence Embeddings for Few-Shot Classification

Author(s): Yongxin Huang, Kexin Wang, Sourav Dutta, Raj Nath Patel, Goran Glavaš, Iryna Gurevych
Paper: https://aclanthology.org/2023.emnlp-main.208.pdf
Code: https://github.com/UKPLab/AdaSent
Model(s): DistilRoBERTa
Dataset(s): SNLI, MultiNLI, Sentence Compression, StackExchange

* Few-shot sentence classification based on pre-trained Sentence Encoders (SE) is efficient
* We investigate strategies for domain-specialization in the context of few-shot sentence classification with SEs
* SetFit achieves strong performance in few-shot classification by contrastively fine-tuning pre-trained sentence embeddings
* SetFit is much more efficient than popular prompt-based methods including In-Context Learning and Pattern Exploit Training
* Domain-Adaptive Pre-Training (DAPT) on a vanilla PLM with unlabeled in-domain data can significantly improve its downstream performance
* Refer to the Task-Adaptive Pre-Training (TAPT)
* AdaSent, combines DAPT and SEPT in a modular fashion
* SetFit is a two-step training procedure based on pre-trained sentence-embedding Transformer models for few-shot sentence classification

#### D5-5 Ditto: A Simple and Efficient Approach to Improve Sentence Embeddings

Author(s): Qian Chen, Wen Wang, Qinglin Zhang, Siqi Zheng, Chong Deng, Hai Yu, Jiaqing Liu, Yukun Ma, Chong Zhang
Paper: https://aclanthology.org/2023.emnlp-main.359.pdf
Code: https://github.com/alibaba-damo-academy/SpokenNLP/tree/main/ditto
Model(s): BERT, Ditto
Dataset(s): NLI, STS

* The anisotropy problem in sentence representations from pre-trained language models, e.g., BERT, without fine-tuningdegrees
* Sentence embeddings from BERT suffer from a bias towards uninformative words
* We propose a simple and efficient unsupervised approach, Diagonal Attention Pooling (Ditto), which weights words with model-based importance estimations and computes the weighted average of word representations from pre-trained models as sentence embeddings

#### D5-6 Connecting degree and polarity: An artificial language learning study

Author(s): Lisa Bylinina, Alexey Tikhonov, Ekaterina Garmash
Paper: https://aclanthology.org/2023.emnlp-main.938.pdf
Code: https://github.com/altsoph/artificial_degree_modifiers
Model(s): BERT
Dataset(s): CapitolWords

* A new linguistic generalisation in pre-trained LM
* Focus on degree modifiers (expressions like slightly, very, rather, extremely) and test the hypothesis that the degree expressed by a modifier (low, medium or high degree) is related to the modifier’s sensitivity to sentence polarity (whether it shows preference for affirmative or negative sentences or neither)

#### D5-7 LLM: The Need for Nuance in Current Debates and a Pragmatic Perspective on Understanding

Author(s): Bram Van Dijk, Tom Kouwenhoven, Marco Spruit, Max Johannes van Duijn
Paper: https://aclanthology.org/2023.emnlp-main.779.pdf

* LLM are unparalleled in their ability to generate grammatically correct, fluent text
* Three critiques of LLM capacities: i) that LLM only parrot statistical patterns in the training data; ii) that LLM master formal but not functional language competence; and iii) that language learning in LLM cannot inform human language learning

#### D5-8 Conversation Understanding using Relational Temporal Graph Neural Networks with Auxiliary Cross-Modality Interaction

Author(s): Cam Van Thi Nguyen, Tuan Anh Mai, Son Le The, Dang Hai Kieu, Duc-Trong Le
Paper: https://aclanthology.org/2023.emnlp-main.937.pdf
Model(s): CORECT
Dataset(s): IEMOCAP, CMU-MOSEI

* Emotion recognition is a crucial task for human conversation understanding
* We propose the Relational Temporal Graph Neural Network with Auxiliary Cross-Modality Interaction (CORECT), an novel neural network framework that effectively captures conversation-level cross-modality interactions and utterance-level temporal dependencies with the modality-specific manner for conversation understanding
* Propose a COnversation understanding model using RElational Temporal Graph Neural Network with Auxiliary Cross-Modality Interaction (CORECT)

#### D5-9 LLM Only Pass Primary School Exams in Indonesia: A Comprehensive Test on IndoMMLU

Author(s): Fajri Koto, Nurul Aisyah, Haonan Li, Timothy Baldwin
Paper: https://aclanthology.org/2023.emnlp-main.760.pdf
Code: https://github.com/fajri91/IndoMMLU
Model(s): GPT-3.5, XGLM, Falcon, BLOOMZ, mT0, LLaMA, Bactrian-X-LLaMA
Dataset(s): IndoMMLU

* LLM are pre-trained on large-scale multilingual texts and their reasoning abilities and real-world knowledge are mainly evaluated based on English datasets
* We introduce IndoMMLU, the first multi-task language understanding benchmark for Indonesian culture and languages, which consists of questions from primary school to university entrance exams in Indonesia

## Poster Stalls

#### D5-10 A Rose by Any Other Name would not Smell as Sweet: Social Bias in Names Mistranslation

Author(s): Sandra Camille Sandoval, Jieyu Zhao, Marine Carpuat, Hal Daumé III
Paper: https://aclanthology.org/2023.emnlp-main.239.pdf
Model(s): OPUS (Marian), Google Translate, Microsoft Translator
Dataset(s): DNIC

* Common models translate Alicia to Alice which is not correct
* Odds of mistranslation are higher for certain social groups (female, black)
* They have a “Diverse Names In Context” dataset which helps in identifying with overlooked issues

#### D5-11 WSP-NMT: Code-switching with Word Senses for Pre-training in Neural Machine Translation

Author(s): Vivek Iyer, Edoardo Barba, Alexandra Birch, Jeff Z. Pan, Roberto Navigli
Paper: https://aclanthology.org/2023.findings-emnlp.859.pdf
Model(s): ESCHER, AMuSEWSD
Dataset(s): XL-WSD

* The word “run” works in three different senses in “Run a marathon”, “Run a mill” and “Run for election”
* Use of KG which informs translation
* Not sure how KG is derived using Word Sense Disambiguation

#### D5-12 Salespeople vs Salesbot: Exploring the Role of Educational Value in Conversational Recommender Systems

Author(s): Lidiya Murakhovs'ka, Philippe Laban, Tian Xie, Caiming Xiong, Chien-Sheng Wu
Paper: https://aclanthology.org/2023.findings-emnlp.657.pdf
Code: https://github.com/salesforce/salesbot
Model(s): sentence-transformers/all-mpnet-base-v2, ChatGPT, GPT3 text-ada-001, Action Decision module, rule-based system, Query Generation, keyword method, Regeneration
Dataset(s): COOKIE, Amazon Product Reviews

* This system can read Buyer Guide and Product Catalog. During conversations, it can make use data from the guide and make suggestions from the catalog 
* The match between what the statement and preferences is made using cosine similarity

#### D5-13 Compressing Context to Enhance Inference Efficiency of LLM

Author(s): Yucheng Li, Bo Dong, Frank Guerin, Chenghua Lin
Paper: https://aclanthology.org/2023.emnlp-main.391.pdf
Code: https://github.com/salesforce/salesbot
Model(s): GPT-3.5, GPT-4, LLaMA-7B, 13B, 30B, Vicuna-7B, 13B
Dataset(s): BBC News, arXiv Articles, ShareGPT.com

* Reduce computation costs by reducing the tokens in the context
* Remove less important information similar to stop word removal
* Removal is done by using a compression ratio: more compression, less accuracy and less compression, more accuracy

#### D5-14 Parameter-efficient Tuning for LLM without Calculating its Gradients

Author(s): Feihu Jin, Jiajun Zhang, Chengqing Zong
Paper: https://aclanthology.org/2023.emnlp-main.22.pdf
Model(s): T5, GPT-2

* Completely bypass backprop based tuning
* PEFT reduces memory needed but is still backprop
* A “bridge model” adapts the LLM and the parameter efficient
* How the dimensionality mismatch is resolved is not clear

#### D5-15 Universal Self-Adaptive Prompting

Author(s): Xingchen Wan, Ruoxi Sun, Hootan Nakhost, Hanjun Dai, Julian Martin Eisenschlos, Sercan O Arik, Tomas Pfister
Paper: https://aclanthology.org/2023.emnlp-main.461.pdf
Model(s): PaLM-62B, PaLM-540B
Dataset(s): commonsense reasoning: {winogrande, piqa, storycloze, boolq, copa, wsc, arc_e, arc_c},  NLI: {anlir1, anlir2, anlir3, rte},
context comprehension: {wic}, 
reading comprehension MCQ: {raceh, racem},
word completion cloze: {lambada},
open-domain QA: {web_questions, natural_questions, triviaqa_wiki},
reading comprehension QA: {squad},
summarization: {xsum, wikilingua}

#### D5-16 A study on Accessing Linguistic Information in Pretrained LM by Using Prompts

Author(s): Marion Di Marco, Katharina Hämmerl, Alexander Fraser
Paper: https://aclanthology.org/2023.emnlp-main.454.pdf
Code: https://github.com/timoschick/pet
Model(s): Pattern-Exploiting Training (PET)
Dataset(s): Universal Dependency Treebank

* Try to extract attributes like gender, tense, case from LLM
* Work done in German, Icelandic and Spanish
* Pattern Exploitation is a very important concept for us to understand

#### D5-17 Our X-ray for LLM

Author(s): Shahar Katz, Yonatan Belinkov

* Rare visualization tool to look “inside” any Transformer based LLM
* MUST try this out because if this works, it will be a good tool to get insights

#### D5-18 CONTRASTE: Supervised Contrastive Pretraining with Aspect-based Prompts for Aspect Sentiment Triplet Extraction

Author(s): Rajdeep Mukherjee, Nithish Kannen, Saurabh Kumar Pandey, Pawan Goyal
Paper: https://aclanthology.org/2023.findings-emnlp.807.pdf
Code: https://github.com/nitkannen/CONTRASTE/
Model(s): CONTRASTE-MTL (T5)
Dataset(s): Lap14, 14Res, 15Res, 16Res, ASTE-Data-V2

* Note “contrastive learning” has been used. See t-SNE above
* Can we also do similarly in our custom embeddings task?

#### D5-19 Enabling LLM to Generate Text with [Citations]

Author(s): Tianyu Gao, Howard Yen, Jiatong Yu, Danqi Chen
Paper: https://aclanthology.org/2023.emnlp-main.398.pdf
Code: https://github.com/princeton-nlp/ALCE
Model(s): InstructGPT (text-davinci-003), TRUE ( T5-11B)
Dataset(s): ASQA, QAMPARI, ELI5

* Read text from a corpus along with citation. This make traceability and explainability possible

#### D5-20 Synthetic Data Generation with LLM for Text Classification: Potential and Limitations

Author(s): Zhuoyan Li, Hangxiao Zhu, Zhuoran Lu, Ming Yin
Paper: https://aclanthology.org/2023.emnlp-main.647.pdf
Model(s): GPT3.5-Turbo, BERT, RoBERTa
Dataset(s): AG’s news, IMDB reviews, SMS spam, Financial phrase bank, Reddit emotion, Relation classification, Tweet irony speech, Tweet emotions, Sarcasm news, Humor speech

* Collecting data is costly
* LLM are data hungry
* Use LLM to generate data by well-crafted Prompt

#### D5-21 Improving Diversity of Demographic Representation in LLM via Collective-critiques and Self-voting

Author(s): Preethi Lahoti, Nicholas Blumm, Xiao Ma, Raghavendra Kotikalapudi, Sahitya Potluri, Qijun Tan, Hansa Srinivasan, Ben Packer, Ahmad Beirami, Alex Beutel, Jilin Chen
Paper: https://aclanthology.org/2023.emnlp-main.643.pdf
Model(s): Flan-PaLM 540B
Dataset(s): People-diversity, Cultural-diversity

* Bring in diversity to responses by creation of multiple reasoning paths

#### D5-22 Aligning Predictive Uncertainty with Clarification Questions in Grounded Dialog

Author(s): Kata Naszadi, Putra Manggala, Christof Monz
Paper: https://aclanthology.org/2023.findings-emnlp.999.pdf
Code: https://github.com/naszka/uncertain_builder
Model(s): T5-small
Dataset(s): Minecraft Dialogue Corpus (MDC), IGLU-NLP, IGLU-MULTI

* Creation of a world in Minecraft. Looked somewhat similar to Winograds work

#### D5-23 LLM Meet Harry Potter: A Bilingual Dataset for Aligning LLM with Characters

Author(s): Nuo Chen, Yan Wang, Haiyun Jiang, Deng Cai, Yuhan Li, Ziyang Chen, Longyue Wang, Jia Li
Paper: https://aclanthology.org/2023.findings-emnlp.570.pdf
Code: https://nuochenpku.github.io/HPD.github.io
Model(s): Alpaca, ChatGLM-6B, GPT3, ChatGPT, ChatGLM
Dataset(s): Harry Potter Dialogue (HPD)

* Harry’s response is very positive in the default ChatGPT and GPT-4
* Using their dataset, Harry responds bit negatively
* EN and CN
* See link

#### D5-24 EffEval: A Comprehensive Evaluation of Efficiency for MT Evaluation Metrics

Author(s): Daniil Larionov, Jens Grünwald, Christoph Leiter, Steffen Eger
Paper: https://aclanthology.org/2023.findings-emnlp.7.pdf
Code: https://github.com/NL2G/effeval
Model(s): DistilBERT, TinyBERT, multilingual DistilMBERT
Dataset(s): WMT15, WMT16, WMT21

* MT evaluation metrics 
* COMET is computationally very light compared to traditionally methods used in LLM

#### D5-25 Theory of Mind for Multi-Agent Collaboration via LLM

Author(s): Huao Li, Yu Quan Chong, Simon Stepputtis, Joseph Campbell, Dana Hughes, Charles Michael Lewis, Katia P. Sycara
Paper: https://aclanthology.org/2023.emnlp-main.13.pdf
Model(s): GPT-4, ChatGPT, GPT-4 + Belief, MAPPO, CBS Planner

* Theory of mind refers to the capacity to understand other people by ascribing mental states to them
* Use of LLM in team configuration to find a bomb hidden in a room
* Theory-of-Mind tasks to several LLMs, concluding that current models (e.g., GPT-4) perform comparably to 9-year old children (Kosinski, 2023). However, the research community has expressed doubts about the validity of text-based ToM tests on machine intelligence(Ullman, 2023; Sap et al., 2023)
* Lim et al. (2020) introduced a method to integrate Bayesian Theory of Mind (BToM) (Baker et al., 2017) with optimal-planning agents in a cooperative game

#### D5-26 Creator Context for Tweet Recommendation

Author(s): Spurthi Amba Hombaiah, Tao Chen, Mingyang Zhang, Michael Bendersky, Marc Najork, Matt Colen, Sergey Levi, Vladimir Ofitserov and Tanvir Amin
Paper: https://aclanthology.org/2023.emnlp-industry.34.pdf
Model(s): AdaBoost, LDA, BERT
Dataset(s): MS-MARCO,   Krestel et al. (2015)  derived from Signal1M, custom Tweets

* LLM cannot figure out possessive determiners and pronouns
* Having this information can make life easy for journalists
* See better Tweets picked for a local story with this system rather than otherwise


#### D5-27 JarviX: An LLM No code Platform for Tabular Data Analysis and Optimization

Author(s): Shang-Ching Liu, ShengKun Wang, Tsungyao Chang, Wenqi Lin, Chung-Wei Hsiung, Yi-Chen Hsieh, Yu-Ping Cheng, Sian-Hong Luo and Jianwei Zhang
Paper: https://aclanthology.org/2023.emnlp-industry.59.pdf
Model(s): vicuna-13b1.1-gptq-4bit-128g

* Get the best biz insights and recommendations for a Solar Cell factory

#### D5-28 Logic-LM: Empowering LLM with Symbolic Solvers for Faithful Logical Reasoning

Author(s): Liangming Pan, Alon Albalak, Xinyi Wang, William Yang Wang
Paper: https://arxiv.org/pdf/2305.12295.pdf
Code: https://github.com/teacherpeterpan/Logic-LLM
Model(s): LOGIC-LM implemented in: gpt-3.5-turbo, text-davinci-003, gpt-4
Dataset(s): PrOntoQA, ProofWriter, FOLIO, LogicalDeduction, AR-LSAT

* Neuro-symbolic reasoning
* Given a problem problem is formulated in FOL, Logic Programming, Constraint Optimization or SMT Solver whichever one is appropriate for the problem

#### D5-29 ChatCoT: Tool-Augmented Chain-of-Thought Reasoning on Chat-based LLM

Author(s): Zhipeng Chen, Kun Zhou, Beichen Zhang, Zheng Gong, Xin Zhao, Ji-Rong Wen
Paper: https://aclanthology.org/2023.findings-emnlp.985.pdf
Code: https://github.com/RUCAIBOX/ChatCoT
Model(s): ChatGPT, GPT-3, PaLM, LLaMA, Galactica, Minerva, PaLM 2
Dataset(s): MATH, HotpotQA

* Decompose a complex problem into multiple subproblems. Solve each one individually with the most appropriate tools
* Good idea



