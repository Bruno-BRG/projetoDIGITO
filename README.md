# 🧠 MNIST Digit Recognition - Aprendizado por Reforço Interativo

Sistema de reconhecimento de dígitos manuscritos com **aprendizado contínuo baseado em feedback do usuário**.

## 🚀 Funcionalidades Principais

### ✨ **Sistema de Aprendizado por Reforço**
- **Feedback Interativo**: Corrija predições erradas em tempo real
- **Fine-tuning Automático**: Modelo aprende com seus erros
- **Estatísticas de Aprendizado**: Acompanhe a evolução do modelo
- **Persistência de Dados**: Feedback salvo automaticamente

### 🎯 **Interface Simplificada**
- **3 Abas Principais**:
  1. **Reconhecimento + Feedback**: Desenhe e corrija predições
  2. **Estatísticas de Aprendizado**: Métricas e fine-tuning
  3. **Treinamento Inicial**: Treinamento opcional do zero

### 🖼️ **Reconhecimento Visual**
- Canvas de desenho interativo
- Preprocessamento automático (28x28)
- Predições com nível de confiança
- Feedback visual por cores

## 🛠️ Instalação

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar aplicação
python main.py
```

## 📖 Como Usar

### 1. **Reconhecimento com Feedback**
1. Abra a aba "Reconhecimento + Feedback"
2. Desenhe um dígito no canvas
3. Clique em "Reconhecer"
4. Se a predição estiver errada:
   - Clique em "ERROU"
   - Selecione o dígito correto
   - Confirme a correção

### 2. **Aprendizado Contínuo**
1. Vá para a aba "Estatísticas de Aprendizado"
2. Visualize métricas e gráficos
3. Clique em "Executar Fine-tuning" para treinar com feedback
4. Acompanhe a melhoria da acurácia

### 3. **Treinamento Inicial (Opcional)**
- Use a aba "Treinamento Inicial" se quiser treinar um modelo do zero
- O sistema já vem com um modelo pré-treinado

## 🧪 Testes

Execute os seguintes testes para validar o sistema:

```bash
# Teste básico
python test_quick.py

# Teste completo do sistema de RL
python test_rl_system.py
```

## 📁 Estrutura de Arquivos

```
projetoDIGITO/
├── main.py              # Interface principal com RL
├── model.py             # Arquitetura CNN
├── trainer.py           # Treinamento e fine-tuning
├── mnist_model.pth      # Modelo pré-treinado
├── requirements.txt     # Dependências
├── user_feedback.json   # Dados de feedback (gerado automaticamente)
├── test_quick.py        # Teste básico
├── test_rl_system.py    # Teste do sistema RL
└── data/MNIST/          # Dataset MNIST
```

## 🎯 Sistema de Feedback

### Como Funciona:
1. **Feedback Positivo**: Quando você marca "ACERTOU", o sistema salva a predição correta
2. **Feedback Negativo**: Quando você marca "ERROU" e corrige, o sistema aprende com o erro
3. **Fine-tuning**: O modelo é re-treinado com os dados de feedback
4. **Persistência**: Todos os feedbacks são salvos em `user_feedback.json`

### Estatísticas Disponíveis:
- **Acurácia Geral**: % de acertos desde o início
- **Acurácia Recente**: % de acertos nos últimos 20 exemplos
- **Distribuição de Erros**: Gráfico dos tipos de erro mais comuns
- **Evolução Temporal**: Como a acurácia melhora com o tempo

## 🔧 Arquitetura Técnica

### Classes Principais:
- **`FeedbackManager`**: Gerencia persistência e estatísticas de feedback
- **`MNISTTrainer`**: Treina modelo e executa fine-tuning
- **`DrawingCanvas`**: Canvas de desenho com preprocessamento
- **`MNISTApp`**: Interface principal com 3 abas

### Métodos de Fine-tuning:
- **`fine_tune_with_feedback()`**: Re-treina modelo com dados de feedback
- **`validate_on_feedback()`**: Valida melhorias na acurácia

## 🎨 Interface Visual

### Canvas de Desenho:
- **Tamanho**: Ajustável
- **Pincel**: Configurável
- **Limpeza**: Um clique

### Feedback Visual:
- **Verde**: Alta confiança (>80%)
- **Laranja**: Média confiança (50-80%)
- **Vermelho**: Baixa confiança (<50%)

### Gráficos e Estatísticas:
- **Gráfico de Pizza**: Distribuição de acertos/erros
- **Métricas Numéricas**: Estatísticas detalhadas
- **Progresso**: Barra de progresso durante fine-tuning

## 📊 Exemplo de Uso

```
Usuário desenha "8" → Sistema prediz "3" → Usuário clica "ERROU" → 
Seleciona "8" → Confirma → Sistema salva feedback → 
Executa fine-tuning → Melhora predições futuras
```

## 🚀 Próximos Passos

- [ ] Implementar aprendizado ativo (sugestão de exemplos difíceis)
- [ ] Adicionar múltiplos modelos comparativos
- [ ] Exportar/importar conjuntos de feedback
- [ ] Análise de padrões de erro
- [ ] Interface web complementar

---

**🎯 Objetivo**: Demonstrar como sistemas de ML podem aprender continuamente com feedback humano, melhorando suas predições de forma interativa e transparente.