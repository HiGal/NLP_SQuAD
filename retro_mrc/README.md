# Retrospective Reader for Machine Reading Comprehension

This repo is our summary and playground for MRC.  More features are coming.

In this work, MRC model is regarded as two-stage Encoder-Decoder architecture. Our main attempts are shared in this repo. 


### Decoder:

The implementation is based on [Transformers](https://github.com/huggingface/transformers) 

#### Retrospective Reader

1) train a sketchy reader (`sh_albert_cls.sh`)

2) train an intensive reader (`sh_albert_av.sh`)

3) rear verification: merge the prediction for final answer (`run_verifier.py`)

Our result SQuAD 2.0 Dev Results for RR with Albert base v2:	

```
{
  "exact": 78.30371430977848,
  "f1": 81.62395967276642,
  "total": 11873,
  "HasAns_exact": 74.34210526315789,
  "HasAns_f1": 80.99211761045146,
  "HasAns_total": 5928,
  "NoAns_exact": 82.25399495374263,
  "NoAns_f1": 82.25399495374263,
  "NoAns_total": 5945
}
```

Initial code was taken and udapted from [AwesomeMRC](https://github.com/cooelf/AwesomeMRC) 
### Citation

```
@article{zhang2020retrospective,
  title={Retrospective reader for machine reading comprehension},
  author={Zhang, Zhuosheng and Yang, Junjie and Zhao, Hai},
  journal={arXiv preprint arXiv:2001.09694},
  year={2020}
}
```

