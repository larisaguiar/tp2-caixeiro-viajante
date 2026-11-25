Define um modelo LaTeX para o texto de Prática de Pesquisa. O projeto possui os seguintes arquivos principais:

- **main.tex**: Arquivo principal do texto;
- **references.bib**: Exemplo de base bibliográfica;
- **apalike-pt.bst**: Estilo bibliográfico em português, usado por padrão;
- **apalike.bst**: Estilo bibliográfico em inglês, para uso alternativo;
- **sbc-template.sty** e **sbc.bst**: Arquivos do template base.

Por padrão, o template utiliza o estilo bibliográfico em português:

```latex
\bibliographystyle{apalike-pt}
```

Caso seja necessário usar o estilo em inglês, basta alterar para:

```latex
\bibliographystyle{apalike}
```

Essa configuração fica no final do arquivo `main.tex`.
