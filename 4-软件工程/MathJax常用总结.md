## LaTex常用方法总结

行内公式用\$...\$， 公式行用\$\$...\$\$。

```mathematica
x = {-b \pm \sqrt{b^2-4ac} \over 2a}
```

$$
x = {-b \pm \sqrt{b^2-4ac} \over 2a}
$$



```mathematica
\sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}
```

$$
\sum_{i=0}^n i^2 = \frac{(n^2+n)(2n+1)}{6}
$$



```mathematica
f(a) = \frac{1}{2\pi i} \oint\frac{f(z)}{z-a}dz
```

$$
f(a) = \frac{1}{2\pi i} \oint\frac{f(z)}{z-a}dz
$$



```mathematica
\sigma = \sqrt{ \frac{1}{N} \sum_{i=1}^N (x_i -\mu)^2}
```

$$
\sigma = \sqrt{ \frac{1}{N} \sum_{i=1}^N (x_i -\mu)^2}
$$

```mathematica
\cos(θ+φ)=\cos(θ)\cos(φ)−\sin(θ)\sin(φ)
```

$$
\cos(θ+φ)=\cos(θ)\cos(φ)−\sin(θ)\sin(φ)
$$



```mathematica
\int_D ({\nabla\cdot} F)dV=\int_{\partial D} F\cdot ndS
```

$$
\int_D ({\nabla\cdot} F)dV=\int_{\partial D} F\cdot ndS
$$



```mathematica
\vec{\nabla} \times \vec{F} = \left( \frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z} \right) \mathbf{i}  + \left( \frac{\partial F_x}{\partial z} - \frac{\partial F_z}{\partial x} \right) \mathbf{j}
```

$$
\vec{\nabla} \times \vec{F} =   \left( \frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z} \right) \mathbf{i}         + \left( \frac{\partial F_x}{\partial z} - \frac{\partial F_z}{\partial x} \right) \mathbf{j}
$$



```mathematica
(\nabla_X Y)^k = X^i (\nabla_i Y)^k = X^i \left( \frac{\partial Y^k}{\partial x^i} + \Gamma_{im}^k Y^m \right)
```

$$
(\nabla_X Y)^k = X^i (\nabla_i Y)^k = X^i \left( \frac{\partial Y^k}{\partial x^i} + \Gamma_{im}^k Y^m \right)
$$



**符号表示**

| 样例                                                         | 公式                                                         |
| :----------------------------------------------------------- | ------------------------------------------------------------ |
| $\alpha, \beta, …, \omega$                                   | \alpha, \beta, …, \omega                                     |
| $\Gamma, \Delta, …, \Omega$                                  | \Gamma, \Delta, …, \Omega                                    |
| $\epsilon  \varepsilon$                                      | \epsilon \varepsilon                                         |
| $\phi  \varphi$                                              | \phi \varphi                                                 |
| $\ell$                                                       | \ell                                                         |
| $\lt \gt \le \leq \leqq \leqslant $                          | \lt \gt \le \leq \leqq \leqslant                             |
| $\ge \geq \geqq \geqslant \neq$                              | \ge \geq \geqq \geqslant \neq                                |
| $\times \div \pm \mp \cdot$                                  | \times \div \pm \mp \cdot                                    |
| $\cup \cap \setminus $                                       | \cup \cap \setminus                                          |
| $\subset \subseteq \subsetneq \supset$                       | \subset \subseteq \subsetneq \supset                         |
| $\in \notin \emptyset \varnothing$                           | \in \notin \emptyset \varnothing                             |
| ${n+1 \choose 2k}$                                           | {n+1 \choose 2k}                                             |
| $\binom{n+1}{2k}$                                            | \binom{n+1}{2k}                                              |
| $\lim_{x\to 0}$                                              | \lim_{x\to 0}                                                |
| $sum_{i=0}^\infty i^2$                                       | sum_{i=0}^\infty i^2                                         |
| $\to \rightarrow \leftarrow \Rightarrow \Leftarrow \mapsto \leftrightarrow$ | \to \rightarrow \leftarrow \Rightarrow \Leftarrow \mapsto \leftrightarrow |
| $\land \lor \lnot \forall \exists \top \bot \vdash \vDash$   | \land \lor \lnot \forall \exists \top \bot \vdash \vDash     |
| $\star \ast \oplus \circ \bullet$                            | \star \ast \oplus \circ \bullet                              |
| $\approx \sim \simeq \cong \equiv \prec \lhd \therefore$     | \approx \sim \simeq \cong \equiv \prec \lhd \therefore       |
| $\infty \aleph_0 \nabla \Im \Re$                             | \infty \aleph_0 \nabla  \Im \Re                              |
| $\partial$                                                   | \partial                                                     |
| $a\equiv b\pmod n$                                           | a \equiv b \pmod n                                           |
| $a_1,a_2,\ldots,a_n$                                         | a_1,a_2,\ldots,a_n                                           |
| $a_1+a_2+\cdots+a_n$                                         | a_1+a_2+\cdots+a_n                                           |
| $\Biggl(\biggl(\Bigl(\bigl((x)\bigr)\Bigr)\biggr)\Biggr)$    | \Biggl(\biggl(\Bigl(\bigl((x)\bigr)\Bigr)\biggr)\Biggr)      |
| $\lceil x \rceil$                                            | \lceil x \rceil                                              |
| $\lfloor x \rfloor$                                          | \lfloor x \rfloor                                            |
| $\Vert{x}\Vert$                                              | \Vert x \Vert                                                |
| $\vert x \vert$                                              | \vert x \vert                                                |
| $\hat{X}$                                                    | \hat{X}                                                      |
| $\bar{X}$                                                    | \bar{X}                                                      |
| $\tilde X$                                                   | \tilde{X}                                                    |
| $\langle{x}\rangle$                                          | \langlex\rangle                                              |
| $\left.\frac12\right\rbrace$                                 | \left.\frac12\right\rbrace                                   |
| $\backslash$                                                 | \backslash                                                   |
| $a\\b$                                                       | because `\\` is for a new line.                              |
| 空白                                                         | \quad                                                        |

 

**希腊字母表示**

|               |             |             |           |          |          |
| ------------- | ----------- | ----------- | --------- | -------- | -------- |
| $\alpha$      | \alpha      | $\omega$    | \omega    | $\kappa$ | \kappa   |
| $\beta$       | \beta       | $\psi$      | \psi      | $\pi$    | \pi      |
| $\gamma$      | \gamma      | $\chi$      | \chi      | $\phi$   | \phi     |
| $\delta$      | \delta      | $\rho$      | \rho      | $\sigma$ | \sigma   |
| $\lambda$     | \lambda     | $\epsilon$  | \epsilon  | $\theta$ | \theta   |
| $\varrho$     | \varrho     | $\zeta$     | \zeta     | $$       | \upsilon |
| $\varepsilon$ | \varepsilon | $\mu$       | \mu       | $\xi$    | \xi      |
| $\varkappa$   | \varkappa   | $\nu$       | \nu       | $\tau$   | \tau     |
| $\varpi$      | \varpi      | $\varsigma$ | \varsigma | $\iota$  | \iota    |
| $\varphi$     | \varphi     | $\vartheta$ | \vartheta | $\eta$   | \eta     |

​       

  **更改字体**

- Use `\mathbb` or `\Bbb` for "blackboard bold": $\mathbb{ABCDEFG}$
- Use `\mathbf` for boldface: $\mathbf{ABCDEFG}$
- Use `\mathit` for italics: $\mathit{ABCDEFG}$
- Use `\pmb` for boldfaced italics: $\pmb{ABCDEFG}$
- Use `\mathtt` for "typewriter" font: $\mathtt{ABCDEFG}$
- Use `\mathrm` for roman font: $\mathrm{ABCDEFG}$
- Use `\mathsf` for sans-serif font: $\mathsf{ABCDEFG}$
- Use `\mathcal` for "calligraphic" letters: $\mathcal{ABCDEFG}$
- Use `\mathscr` for script letters: $\mathit{ABCDEFG}$
- Use `\mathfrak` for "Fraktur" (old German style) letters: $\mathfrak{ABCDEFG}$



参考文档地址：https://pic.plover.com/MISC/symbols.pdf

http://garsia.math.yorku.ca/MPWP/LATEXmath/latexsym.html

 

![img](http://garsia.math.yorku.ca/MPWP/LATEXmath/arrow1.gif)

![img](http://garsia.math.yorku.ca/MPWP/LATEXmath/relation1.gif)

![img](http://garsia.math.yorku.ca/MPWP/LATEXmath/relation2.gif)

