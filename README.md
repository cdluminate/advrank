Adversarial Ranking Attack and Defense (ECCV2020)
===

Materials for ECCV-2020 Paper #2274.

* **Title:** Adversarial Ranking Attack and Defense
* **Authors:** Mo Zhou, Zhenxing Niu, Le Wang, Qilin Zhang, Gang Hua
* **Preprint:** [https://arxiv.org/abs/2002.11293](https://arxiv.org/abs/2002.11293)
* **Full-Preprint:** [2002.11293v3-full.pdf (with more graphs)](assets/2002.11293v3-full.pdf)
* **Video:** [{Bilibili(English)}](https://www.bilibili.com/video/BV1ih411Z7Rn/) [{Bilibili(Chinese)}](https://www.bilibili.com/video/BV1fZ4y1T7oC/) [{Youtube(English)}](https://www.youtube.com/watch?v=vlxfQ-Ip1lQ)
* **Slides:** [(One-page version)](assets/eccv20-short.pdf) [(long version)](assets/eccv20-long.pdf)
* **Code:** [available here](https://github.com/cdluminate/advrank/tree/master/Code) (Released on Aug 24)

## Demonstration

<table>
  <tr>
    <td>
      <img src="assets/s1.png"/>
    </td>
    <td>
      <img src="assets/s3.png"/>
    </td>
  </tr>
  <tr>
    <td>
      <img src="assets/s4.png"/>
    </td>
    <td>
      <img src="assets/s5.png"/>
    </td>
  </tr>
  <tr>
    <td>
      <img src="assets/s6.png"/>
    </td>
    <td>
      <img src="assets/s7.png"/>
    </td>
  </tr>
</table>

## Contributions

Definition of *Adversarial ranking attack*: adversarial ranking attack aims
*raise* or *lower* the ranks of some chosen candidates C={c₁,c₂, ... ,cₘ} with
respect to a specific query set Q={q₁,q₂, ... ,qw}.  This can be achieved by
either Candidate Attack (CA) or Query Attack (QA).

1. The adversarial ranking attack is defined and implemented, which can
intentionally change the ranking results by perturbing the candidates
or queries.

2. An adversarial ranking defense method is proposed to improve the
ranking model robustness, and mitigate all the proposed attacks
simultaneously.

## License and Bibtex

The paper (PDF file) is distributed under the [CC BY-SA-NC 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

The code is published under the [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

Bibtex for the ECCV version:
```latex
@InProceedings{advrank,
  author={Zhou, Mo and Niu, Zhenxing and Wang, Le and Zhang, Qilin and Hua, Gang},
  title={Adversarial Ranking Attack and Defense},
  booktitle={ECCV 2020},
  year={2020},
  pages={781--799},
  isbn={978-3-030-58568-6}
}
```

Bibtex for the ArXiv preprint version:
```latex
@article{zhou2020advrank,
  title={Adversarial Ranking Attack and Defense},
  author={Zhou, Mo and Niu, Zhenxing and Wang, Le and Zhang, Qilin and Hua, Gang},
  journal={arXiv preprint arXiv:2002.11293},
  year={2020}
}
```

## References

1. [A. Madry et.al. Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)
