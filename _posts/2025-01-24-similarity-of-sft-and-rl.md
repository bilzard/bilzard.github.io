---
layout: post
title: DeepSeek-R1はなぜ高いベンチマーク性能を実現できたのか？
date: 2025-01-24 14:34:33 +0900
categories: blog
tags: LLM DeepSeek-R1 RL
latex: true
toc: true
---

2025年1月22日に公開されたDeepSeek-R1の論文によると、彼らのモデルはProcess Reward ModelやMonte-Carlo Tree Searchのような高度な学習手法を使わずに
OpenAI o1に匹敵するような高いベンチマーク性能を実現しました。本記事ではDeepSeek-R1で用いられた強化学習手法を関連論文とともに整理しながら、
同モデルが高いベンチマーク性能を実現できた要因について考察を試みます。

## 背景1: DeepSeek-R1について

具体的な内容に入る前に、DeepSeek-R1の位置付けと学習手法の特徴について簡単に説明しておきます。

### DeepSeek-R1とは？

DeepSeek-R1は、中国のベンチャー企業DeepSeek-AIによって開発された、論理的推論能力を強化した最新モデルです。
彼らの論文[DeepSeek-R1]によれば、コンペティションレベルの数学のベンチマークである[AIME]や、
実践的なコーディングのベンチマークである[SWE-bench Verified]でOpen-AI o1に匹敵する性能を報告しています。

DeepSeek-R1は実用的なベンチマークで高い性能を報告しているだけでなく、AGIの実現に不可欠と考えられている「(半)自動的なモデルの進化(self-evolution)」を実現した
と主張していることが特筆すべき点です。

![](/images/Screenshot%202025-01-24%20at%2015.29.40.png)

**図は[DeepSeek-R1]より引用。**

### self-evolutionの定義

何をもってself-evolutionを実現したと言えるかについては議論の余地がありますが、ここでは数学の問題のように、
「**問題と最終解答を与えればモデルが問題から最終解答に辿り着くまでの解法を自力で導出できる**」というのを定義とします。

なお、理想的には学習データの生成を完全に自動化することですが、ここでは問題と正答のラベルのペアがあらかじめ与えられていることを前提としているので、「半自動的」という表現を用いました。

### self-evolutionに関する批判的観点

ところで、著者らの主張を鵜呑みにして「DeepSeek-R1はself-evolutionを実現した」と表現するには注意が必要で、
以下のような問題点があります。

1. **Generization** : 学習データに何を使ったのかが公開されていないため、評価データと同一ドメインのデータを大量に使用している可能性があり、**未知の問題に対する汎化性能に疑問が残る**。
2. **Factuality**: 解答の導出過程の正しさについて一切アラインメントを加えていないので、「最終的な解答はあっているが推論過程が誤っている」ケースが含まれる可能性がある。

特に1の観点については慎重に判断する必要があり、仮に評価データの問題と類似の問題が大量に含まれる場合、単純なFine-tuningと区別できないことになり、
「モデルが独力で創発的な学習を実現した」とは言えないはずです。

とはいえ、今までOpenAI o1のようなプロプライエタリモデルでしか解答できなかった難しいタスクにも解答できるモデルを学習した、という意味では実質fine-tuningと同じであっても
画期的なモデルということはできるかもしれません。

以下では上記の批判的観点は一旦傍に置いて、彼らの用いた事後学習手法にスポットを当てて既存手法との違いを見ていくことにします。

## 背景2: 言語モデルの事後学習手法について

強化学習に馴染みのない読者のために、強化学習とSFTについて簡単に補足しておきます。

### 言語モデルの事後学習におけるSFTについて

Supervised Fine-tuning(SFT)は、正解ラベルを与えてモデルのパラメータを学習する手法全般を指しますが、
言語モデル、特に数学タスクのように解答が一意に定まる形式のタスクをとく言語モデルを学習する場合のSFTでは、
以下に示す**Rejection Sampling**と呼ばれる手法が使われることが多いです。

**Rejection Sampling:**

1. ベースとなる生成モデルに1つの問題あたりK個の解答をランダムに生成させる(この時、導出過程も一緒に出力させる)
2. 生成された解答のうち、**正解した解答のみをフィルタリングする**
3. 正解した解答（導出過程を含む）を学習データとしてnext-token predictionタスクで生成モデルをfine-tuningする

このような方法を用いることで、以下のようなメリットがあります。

1. モデルが「正解へと導く解答」を出力しやすくなる
2. （SFT用の）学習データの分布が事前学習データの分布と概ね同じになるので、**事前学習した重みから大きく乖離しにくい**

### 言語モデルの事後学習における強化学習(RL)について

強化学習(Reinforcement Learning; RL)はもともと言語モデルの学習手法とは独立して発展した学習手法ですが、
[RLHF]以降、言語モデルの事後学習手法、特に**人間との対話にアラインする手法**として積極的に用いられるようになりました。

言語モデルの事後学習に用いられるオーソドックスなRL手法の一つとして、[RLHF]でも用いられたProximal Policy Optimization[PPO]があります。

PPOでは以下の3つのモデルを同時、または交互に学習します。

1. **Policy Model**: テキストを生成するモデル
2. **Reward Model**: Policy Modelの生成結果を評価するモデル
3. **Value Function Model**: 生成モデルの途中経過時点での生成内容を評価するモデル

Reward Modelには最終的な生成結果のみ評価するOutput Reward Model(ORM)と、途中の生成結果を逐一評価するProcess Reward Model(PRM)がありますが、
ラベルを用意するのが簡単なORMが用いられるのが一般的です。

### SFTとRLの違いについて

SFTとRLには以下の違いがあります。

1. **学習するモデルの違い**: RLでは生成モデル以外にReward ModelやValue Function Modelを追加で学習する。
2. **目的関数**: SFTでは生成モデルの目的関数としてcross entropyが用いられるが、RLではより保守的な目的関数が用いられる。
3. **Rejection Samplingの有無**: SFTでは「正解へと導くサンプル」のみフィルタリングするのが一般的だが、RLでは「不正解へと導くサンプル」に負の報酬を与えて学習することができる。

まず、学習するモデルがRLの方が多いです。モデルの内部状態についての解釈性は高まりますが、一般的にRLの方が学習コストが高いです。

また、強化学習はSFTに比べて一般的に学習が安定しないので「元のモデルの重みから大幅に乖離しない」ような対策を施した保守的な目的関数が使われるのが一般的です。

他にも、数学タスクのように正解と不正解を機械的に判定できるタスクの場合は「不正解へと導くサンプル」を学習に利用できるかどうかの違いがあります。

RLからReward ModelやValue Functionモデルを除外したものがSFTなので、SFTはRL手法の特殊な場合とみなせます。

## DeepSeek-R1に至るRL手法の発展(Policy GradientからGRPOまで)

以下では、最も基本的なRLの学習手法であるPolicy Gradientを起点として、DeepSeek-R1に用いられるGRPOへと発展する過程を順番に示します。

### SFT

SFTの目的関数はnext-token predictionタスクにおけるcross entropy損失で次のように表せます。

$$
\mathcal{J} ^ \mathrm{SFT} = \frac{1}{\lvert \mathcal{D} _ {k} \rvert} \sum_{\tau \in \mathcal{D} _ {k}} \sum _ {t = 0}^{T} \log \pi _ {\theta}(a _ {t} \vert s _ {t}) R _ k \tag{1}
$$

上記の式で $R _ k$ はRejection Samplingの結果を判定する関数で、正解した場合は $R _ k = 1$, 不正解の場合は $R _ k = 0$ を返します。

### Policy Gradient

最もシンプルなRL学習手法であるPolicy Gradientでは以下のような目的関数を用います。

$$
\mathcal{J} ^ \mathrm{PG} = \frac{1}{\lvert \mathcal{D} _ {k} \rvert} \sum_{\tau \in \mathcal{D} _ {k}} \sum _ {t = 0}^{T} \log \pi _ {\theta}(a _ {t} \vert s _ {t}) \hat{A} _ {t} \tag{2}
$$

$\hat{A} _ t$ はadvantageと呼ばれる量で、value function modelとreward modelの推定値を元に計算される値ですが、計算方法は割愛します。
意味的には「平均的な行動と比較した、ある行動の相対価値」を表します。
この記事の内容を理解するためには「報酬のようなもの」と理解しておけば十分だと思います。実際に報酬をここに直接代入する手法もあるようですが、Advantageを使う方が学習が安定するそうです。

なお、SFTの目的関数(1)と比較すると、報酬関数 $R _ k$ をAdvantage $\hat{A} _ t$ で置き換えた形式になっています。

### PPO

[PPO]の論文によると、Policy Gradientを目的関数とすると勾配の更新が大きくなる傾向にあり、学習が不安的になると記されています。

> While it is appealing to perform multiple steps of optimization on this loss $L^{PG}$ using the same trajectory,
> doing so is not well-justified, and **empirically it often leads to destructively large policy updates**

**[PPO]より引用。**

そこで、PPOではより保守的な目的関数を用います。具体的には以下のいずれか、または両方を用いることができます。

1. Clipped TRPO (eq. 3)
2. KL-divergence Penalty (eq. 4)

$$
\mathcal{J} ^ \mathrm{CLIP} =
        \frac{1}{G} \sum_{k=1}^{G} \min\left(
        \frac{\pi _ {\theta}(o _ k \vert q)}{\pi_{\theta_{\mathrm{old}}}(o _ k \vert q)} \hat{A} _ k,
        \mathrm{clip}\left( \frac{\pi_{\theta}(o _ k \vert q)}{\pi_{\theta_{\mathrm{old}}}(o _ k \vert q)}, 1 - \epsilon, 1 + \epsilon \right) \hat{A} _ k
        \right) \tag{3}
$$

$$
\begin{align}
    \mathcal{D} _ {\mathrm{KL}} \left(\pi _ {\theta} \parallel \pi_{\mathrm{ref}}\right) &=
    \frac{\pi _ {\mathrm{ref}}(o _ k \vert q)}{\pi _ {\theta}(o _ k \vert q)}
    - \log \frac{\pi _ {\mathrm{ref}}(o _ k \vert q)}{\pi _ {\theta}(o _ k \vert q)} - 1 \tag{4} \\
\end{align}
$$

Clipped TRPOではpolicyモデルを直接最適化するのではなく、古いpolicyモデルとの確率分布の相対比を最適化します。
また、古いpolicyモデルとの目的関数の乖離が大きくなりすぎるのを防ぐために$[1-\epsilon, 1+\epsilon]$ の範囲でclippingします。
最終的には報酬のworst caseを反映してclip前の目的関数とclip後の目的関数の最小値を取ります。

また、参照モデルの重みとのKL-divergenceを正則化項に加えることもできます。
なお、[RLHF]ではこちらのケースを採用していて、参照モデルとして1段階目の事後学習が完了した時点の重みを採用しています。

$$
\mathcal{J} _ \mathrm{PPO}(\theta) = \mathbb{E} _ {q \sim \mathcal{P}(Q), \lbrace o _ k \rbrace _ {k=1}^{G} \sim \pi _ {\theta _ {\mathrm{old}}}(O \vert q) }
\left[
    \mathcal{J} _ \mathrm{CLIP} - \beta \mathcal{D} _ {\mathrm{KL}} \left(\pi _ {\theta} \parallel \pi _ {\mathrm{ref}}\right)
\right] \tag{5}
$$

![](/images/Screenshot%202025-01-24%20at%2018.30.59.png)

**図は[PPO]より引用。**

### GRPO

PPOではadvantageの推定にvalue function modelの推定値を用いる必要がありますが、[DeepSeek-Math]で導入されたGRPOでは
問題ごとにサンプリングしたK個の解答の報酬をmean-std正規化した値で代替します(eq. 6)。
これによりvalue functionを学習する必要がなくなります。

$$
\hat{A} _ k = \frac{r _ k - \mathrm{mean}\left(\lbrace r _ 1, r _ 2, \cdots, r _ G \rbrace\right)}{\mathrm{std}\left(\lbrace r _ 1, r _ 2, \cdots, r _ G \rbrace\right)} \tag{6}
$$

また、DeepSeek-R1ではreward modelも無くしてルールベースの報酬（正解した場合を1、それ以外を0）を採用しています。
これによりrewardモデルの学習も不要になりました。

## 考察

### DeepSeek-R1の強化学習はSFTとあまり変わらない

以上で示したように、DeepSeek-R1では学習するモデルがpolicyモデルのみなので、RLフェーズでやっていることは実質的にSFTとほとんど変わらないことがわかります。

両者の違いは以下のような些細なものです。

1. Advantageをバッチ内の解答サンプルの報酬の分布で正規化している
2. 目的関数により保守的な更新ルールを採用している(すでに学習した内容を保ちつつ新たなデータにアラインする)

### 違いをもたらしている要因は何か？

DeepSeek-R1では高度な学習スキームを用いずに冒頭で示したような高いベンチマーク性能を実現したことがわかります。
また、DeepSeek-R1ではOpenAI o1で用いられている可能性が噂されているPRMやMonte-Carlo Tree Search(MCTS)は用いなかったと明言しています。

そうなると、DeepSeek-R1でベンチマーク評価が非常に高い原因は何なのか？が素朴な疑問として湧いてきます。
前述したような学習データの品質については内容が公開されていないので傍に置いておくとして、
それ以外の要因には以下が挙げられます。

1. 保守的な目的関数を採用している
2. 生成モデルのパラメータ数が十分大きいこと
3. 多様な論理的推論用のデータを使って十分なSTEP学習すること

#### 保守的な目的関数を採用している

まず、RLで用いられているGRPOでは元の分布から乖離するのを抑制するような対策を行なっています。
これによって新たなデータに過学習せずにアラインすることが可能だったのかもしれません。

#### 生成モデルのパラメータ数が十分大きいこと

また、生成モデルのパラメータ数が671Bと非常に大きいことも要因の一つかもしれません。
これは著者らの蒸留によるablation studyの結果からも示唆されます。
32BパラメータのQwen2.5をベースとしたモデルに同様のRLを実施しても性能のゲインは得られなかったのに対し、
671Bの巨大モデルにRLを施したモデルを使って推論結果を生成し、同じ32Bパラメータのベースモデルに蒸留した場合は
巨大モデルのゲインを引き継いだと報告しています。
これは著者らの結果を再現するには十分大きなパラメータ数のベースモデルが必要であることを示唆します。

#### 多様な論理的推論用のデータを使って十分なSTEP学習すること

あるいは、学習に用いた論理的推論用データの多様性と学習のSTEP数が関係しているのかもしれません。
過去に学習したDeepSeek-V2の論文[DeepSeek-V2]によると、「論理的推論データは他のデータと異なり、
学習時間を長くするほどパフォーマンスの改善が見られるなど他のデータとは異なる現象が確認できた」としています。
数学やコーディングのデータの学習STEPを他の学習サンプルよりも長くすることが秘訣なのかもしれません。

> In our preliminary experiments, **we find that the RL training on reasoning
> data, such as code and math prompts, exhibits unique characteristics that are distinct from the
> training on general data**. For example, **the mathematical and coding abilities of our model can
> keep improving over a longer period of training steps**.

**[DeepSeek-V2]より引用。**

## 結論

以上、DeepSeek-R1に至るまでのRL目的関数の発展を確認しながら、同モデルがOpenAI o1に匹敵する
高いベンチマーク性能を実現した要因について考察しました。

## 参考文献

- [DeepSeek-R1] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
Reinforcement Learning, DeepSeek-AI, Jun 2025
- [SWE-bench Verified] https://openai.com/index/introducing-swe-bench-verified/
- [AIME] https://maa.org/maa-invitational-competitions/
- [RLHF] Training language models to follow instructions with human feedback, OpenAI, Mar 2022
- [PPO] Proximal Policy Optimization Algorithms, Schulman et. al., Aug 2017
- [Policy Gradient] https://spinningup.openai.com/en/latest/algorithms/vpg.html
- [DeepSeek-Math] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models, DeepSeek-AI et.al, Apr 2024
- [DeepSeek-V2] DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model, DeepSeek-AI, Jun 2024