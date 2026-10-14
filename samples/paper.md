# Retrieval-augmented generation for scientific question answering: a literature review

## Abstract

Gökdemir et al. 2025 report that we survey related work on document parsing, neural information retrieval, metric learning, and retrieval-augmented generation [1]. Wang et al. 2026 report that in this work, we present AstroRAG -- a PageRank-based retrieval-augmented generation (RAG) pipeline adapted for question answering in astronomy [2]. Rathi et al. 2026 report that this work presented a systematic, unified evaluation of retrieval strategies for scientific question answering [3].

## 1. Introduction

This review groups the selected papers into themes. Gökdemir et al. 2025 report that we survey related work on document parsing, neural information retrieval, metric learning, and retrieval-augmented generation [1]. Wang et al. 2026 report that in this work, we present AstroRAG -- a PageRank-based retrieval-augmented generation (RAG) pipeline adapted for question answering in astronomy [2]. Rathi et al. 2026 report that this work presented a systematic, unified evaluation of retrieval strategies for scientific question answering [3].

## 2. Generation, rag, retrieval

Gökdemir et al. 2025 report that we survey related work on document parsing, neural information retrieval, metric learning, and retrieval-augmented generation [1]. Jat et al. 2026 report that in this work, we have developed a Q\&A application inspired by the Retrieval Augmented Generation (RAG), which is comprised of an in-house database indexed on the arXiv articles related to the Electron-Ion Collider (EIC) experiment - one of the largest international scientific collaboration and incorporated an open-source LLaMA model for answer generation [4]. Yu et al. 2026 report that in this work, we propose SF-RAG, an RAG method that introduces the “inherit-and-navigate” paradigm [5].

Rathi et al. 2026 report that their work established the retrieve-then-read framework that forms the foundation of most modern RAG systems [3]. Rathi et al. 2026 report that despite rapid progress in RAG research, limited work systematically compares multiple retrieval pipeline strategies under controlled conditions using large-scale scientific corpora [3]. Rathi et al. 2026 report that recent work has proposed multiple enhancements to traditional retrieval pipelines, including query rewriting, hypothetical document generation, reranking approaches, reciprocal rank fusion methods, self-reflective retrieval systems such as Self-RAG, and agentic retrieval frameworks [3].

Vladika and Matthes 2025 report that we hope our work provides valuable insights for optimizing applied RAG systems in practice [6].

## 3. Retrieval, generation, rag

Wang et al. 2026 report that in this work, we present AstroRAG -- a PageRank-based retrieval-augmented generation (RAG) pipeline adapted for question answering in astronomy [2]. Rathi et al. 2026 report that section 2 reviews related work on RAG systems and retrieval optimization techniques [3]. Gökdemir et al. 2025 report that we survey related work on document parsing, neural information retrieval, metric learning, and retrieval-augmented generation [1].

Hu et al. 2025 report that specifically, for each contextual subgraph ¯G𝑖∈𝑆( ¯G;𝑞), we prompt the LLM to summarize the contextual information surrounding𝑐𝑖 [7]. Yu et al. 2026 report that this section reviews existing methodologies in two key areas relevant to our work: academic paper comprehension systems and retrieval-augmented generation (RAG) techniques [5]. Hu et al. 2025 report that section 2 reviews related work on retrieval-augmented generation [7].

Apurba et al. 2026 report that scientific RAG and QA.RAG combines retrieval with generation for knowledge-intensive NLP (Lewis et al., 2020) [8].

## 4. Scientific, question, answering

Rathi et al. 2026 report that this work presented a systematic, unified evaluation of retrieval strategies for scientific question answering [3]. Lála et al. 2023 report that do not directly answer the question, instead summarize to give evidence to help answer the question [9]. Chen et al. 2026 report that our work relates to three research lines: scientific question answering, retrieval augmented generation with reranking, and agentic scientific QA [10].

Zhiyuan et al. 2025 report that in this work, We addressed the challenge of answering complex scientific questions that entail multiple correlated intents and require evidence from diverse sources and multi-hop reasoning [11]. Gökdemir et al. 2025 report that hiPerRAG delivers robust performance on existing scientific question answering (Q/A) benchmarks and two new benchmarks introduced in this work, achieving 90% accuracy on SciQ and 76% on PubMedQA—outperforming both domain-specific models like PubMedGPT and commercial LLMs such as GPT-4 [1]. Lála et al. 2023 report that we call this PaperQA, an agent-based RAG system for scientific question answering [9].

Liu et al. 2026 report that we present LitTraceQA, a benchmark for literature-grounded question answering over scientific papers [12]. Lála et al. 2023 report that furthermore, PaperQA is competitive to state-of-the-art commercial tools for scientific question answering [9].

## 5. Conclusion

Vladika and Matthes 2025 report that we hope our work provides valuable insights for optimizing applied RAG systems in practice [6]. Apurba et al. 2026 report that scientific RAG and QA.RAG combines retrieval with generation for knowledge-intensive NLP (Lewis et al., 2020) [8]. Lála et al. 2023 report that furthermore, PaperQA is competitive to state-of-the-art commercial tools for scientific question answering [9].

## References

[1] Ozan Gökdemir, Carlo Siebenschuh, Alexander Brace, Azton I. Wells, Brian Hsu, Kyle Hippe, et al. (2025). HiPerRAG: High-Performance Retrieval Augmented Generation for Scientific Insights. Platform for Advanced Scientific Computing Conference (PASC). https://doi.org/10.1145/3732775.3733586
[2] Zhifeng Wang, Jason Jingshi Li, Kaihao Zhang, Ramesh S. Sankaranarayana (2026). AstroRAG -- A Pagerank-Based Retrieval-Augmented Generation Pipeline for Question Answering in Astronomy. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2605.25039
[3] Bhagyesh Rathi, Eshan Chawla, William Andreopoulos (2026). Re-ranking and Late Interaction Drive Retrieval Quality: A Controlled Comparison of RAG Strategies for Scientific Question Answering. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2609.38473
[4] Tina J Jat, T. Ghosh, Karthik Suresh (2026). Retrieval-Augmented Question Answering over Scientific Literature for the Electron-Ion Collider. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2604.02259
[5] Rui Yu, Wang, Tianyi, Ruixia Liu, Yinglong Wang (2026). SF-RAG: Structure-Fidelity Retrieval-Augmented Generation for Academic Question Answering. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2602.13647
[6] Juraj Vladika, Florian Matthes (2025). On the Influence of Context Size and Model Choice in Retrieval-Augmented Generation Systems. Findings of the Association for Computational Linguistics: NAACL. https://doi.org/10.18653/v1/2025.findings-naacl.375
[7] Yuntong Hu, Zhihan Lei, Zhongjie Dai, Allen Zhang, Abhinav Angirekula, Zheng Zhang, et al. (2025). CG-RAG: Research Question Answering by Citation Graph Retrieval-Augmented LLMs. International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR). https://doi.org/10.1145/3726302.3729920
[8] Kaysarul Anas Apurba, Md. Hasibul Hasan, Rofiqul Alam Shehab, Asab Azad (2026). SciRet: A Compute-Aware Empirical Study of Retrieval and Reranking for Scientific RAG. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2608.03860
[9] Jakub Lála, Odhran O’Donoghue, Aleksandar Shtedritski, Sam Cox, Samuel G. Rodriques, Andrew Dickson White (2023). PaperQA: Retrieval-Augmented Generative Agent for Scientific Research. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2312.07559
[10] Haotian Chen, Qingqing Long, Siyu Pu, Xiao Luo, Wei Ju, Meng Xiao, et al. (2026). DeepEra: A Deep Evidence Reranking Agent for Scientific Retrieval-Augmented Generated Question Answering. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2601.16478
[11] Li, Zhiyuan, Yu, Haisheng, Guo, Guangchuan, Zhou, Nan, Zhang, Jiajun (2025). MuISQA: Multi-Intent Retrieval-Augmented Generation for Scientific Question Answering. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2511.16283
[12] Xuye Liu, Yimu Wang, Peng Shi, Bo Xue, Xiangrui Ke, Songcheng Cai, et al. (2026). LitTraceQA: A Benchmark for Multi-Stage Grounding and Verification in Scientific Question Answering. arXiv (Cornell University). https://doi.org/10.48550/arxiv.2608.07370
