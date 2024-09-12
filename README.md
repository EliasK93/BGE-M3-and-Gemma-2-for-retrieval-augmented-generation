## BGE-M3 and Gemma-2 for Retrieval Augmented Generation 

Example application for using the [BGE-M3](https://huggingface.co/BAAI/bge-m3) model and the instruction-tuned 9B variant of Google's [Gemma 2](https://blog.google/technology/developers/google-gemma-2/) model in a Retrieval-Augmented Generation (RAG) pipeline.

As a example use case, all three volumes of the original book versions the _Lord of the Rings_ fantasy novel series are used as source documents for querying during the information retrieval.
Afterwards, some book trivia questions about the _Lord of the Rings_ series are given to the generation model, which it has to answer using the extracted information.

The RAG-Pipeline is constructed using the [LangChain](https://python.langchain.com/v0.2/docs/introduction/) framework, utilizing [FAISS](https://github.com/facebookresearch/faiss) to efficiently search a vector database of the BGE-embedded document chunks and a Hugging Face [pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines) for the response generation.
To be able to use the Gemma-2-Instruct-9B on a consumer GPU, it is loaded in 4-Bit-Mode using a [quantization](https://huggingface.co/docs/accelerate/usage_guides/quantization) approach.

<br>

### Procedure

##### I. Prepare a Vector Database for Retrieval

1. Split each of the the raw txt retrieval source files (LOTR Volumes I, II and III) into overlapping chunks (chunk size = 1000, overlap size = 200).
2. Embed each chunk using the BGE-M3 embedding model.
3. Construct an indexed vector database for the chunks using FAISS.

##### II. Query

1. Embed the raw query (LOTR trivia question) using the BGE-M3 embedding model.
2. Collect the K most relevant chunks to this query (the ones with the most similar vectors) using FAISS.
3. Construct a prompt including the query as question and the collected chunks as context pieces.
4. Run the prompt through the Gemma-2-9B-Instruct generation model and return the model answer.

##### III. Evaluate

1. For each trivia question, compare the true answer and the retrieval-augmented model answer.
2. Mark model answers as correct or incorrect.

<br>

### Results

|                           Trivia Question                            |             Correct Answer              |                                                                             Model Answer                                                                              |     |                                                                                              Used Chunks                                                                                               |
|:--------------------------------------------------------------------:|:---------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|                Which wizard lived in Orthanc?                        |                 Saruman                 |                                                                                Saruman                                                                                |  ✅  |                       The Two Towers (chunk 1793), The Two Towers (chunk 1855), The Fellowship of the Ring (chunk 834), The Two Towers (chunk 1931), The Two Towers (chunk 1401)                       |
|         What was the name of the inn in the village of Bree?         |            The Prancing Pony            |                                                     The inn in the village of Bree was called The Prancing Pony.                                                      |  ✅  | The Fellowship of the Ring (chunk 483), The Fellowship of the Ring (chunk 481), The Fellowship of the Ring (chunk 480), The Fellowship of the Ring (chunk 482), The Fellowship of the Ring (chunk 476) |
|                   What was Gandalf's sword's name?                   |                Glamdring                |                                                        The text does not mention the name of Gandalf's sword.                                                         |  ❌  |  The Fellowship of the Ring (chunk 958), The Fellowship of the Ring (chunk 84), The Fellowship of the Ring (chunk 93), The Fellowship of the Ring (chunk 827), The Fellowship of the Ring (chunk 932)  |
|                         Who married Aragorn?                         |                  Arwen                  |                                                         The text states that Aragorn was in love with Arwen.                                                          |  ✅  |           The Fellowship of the Ring (chunk 792), The Return of the King (chunk 2532), The Two Towers (chunk 1661), The Return of the King (chunk 2536), The Return of the King (chunk 2823)           |
|             Which type of blade was Frodo stabbed with?              |           With a Morgul-knife           |                                                                            A Morgul-knife                                                                             |  ✅  |  The Fellowship of the Ring (chunk 631), The Fellowship of the Ring (chunk 700), The Fellowship of the Ring (chunk 889), The Return of the King (chunk 3073), The Fellowship of the Ring (chunk 630)   |
|                     What food does Gollum like?                      |          Raw fish and rabbits           |                                                                     Gollum likes to eat raw fish.                                                                     | (✅) |                       The Two Towers (chunk 2107), The Two Towers (chunk 2307), The Fellowship of the Ring (chunk 32), The Two Towers (chunk 1998), The Two Towers (chunk 2004)                        |
|                     What was Gollum's real name?                     |                 Sméagol                 |                                                             The text says Gollum's real name was Sméagol.                                                             |  ✅  |                  The Fellowship of the Ring (chunk 32), The Two Towers (chunk 1998), The Two Towers (chunk 2226), The Two Towers (chunk 2278), The Fellowship of the Ring (chunk 168)                  |
| What did Frodo see on the ring after Gandalf threw it into the fire? | Fiery letters in the language of Mordor |Frodo saw fine lines, finer than the finest pen-strokes, running along the ring, outside and inside: lines of fire that seemed to form the letters of a flowing script.| (✅) | The Fellowship of the Ring (chunk 144), The Fellowship of the Ring (chunk 143), The Fellowship of the Ring (chunk 181), The Fellowship of the Ring (chunk 165), The Fellowship of the Ring (chunk 738) |
|                  What was the full name of Pippin?                   |              Peregrin Took              |                                                                             Peregrin Took                                                                             |  ✅  |         The Return of the King (chunk 2610), The Fellowship of the Ring (chunk 654), The Fellowship of the Ring (chunk 1005), The Return of the King (chunk 2454), The Two Towers (chunk 1494)         |
|       Which eagle rescued Gandalf from the tower of Isengard?        |                Gwaihir                  |                                                   Gwaihir the Windlord rescued Gandalf from the tower of Isengard.                                                    |  ✅  |             The Two Towers (chunk 1593), The Fellowship of the Ring (chunk 956), The Fellowship of the Ring (chunk 1058), The Return of the King (chunk 3099), The Two Towers (chunk 1750)             |

<br>

### Requirements

##### - Python >= 3.10

##### - Conda
  - `pytorch==2.4.0`
  - `cudatoolkit=12.1`

##### - pip
  - `transformers`
  - `langchain`
  - `langchain-community`
  - `langchain-huggingface`
  - `faiss-gpu`
  - `accelerate`
  - `bitsandbytes`
  - `peft`
  - `sentencepiece`
  - `protobuf`

<br>

### Notes

The files in this repository which contain text from the books are cut off after the first 50 lines.
