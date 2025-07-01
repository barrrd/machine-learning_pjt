# 🗣️ Korean Multi-Sentence Based Emoji Recommendation and Emotion Recognition System

This project is a Korean emotion recognition system that predicts the user's emotion based on multi-sentence conversational context and recommends suitable emojis accordingly.  
The system considers both single-sentence and **sliding window-based multi-sentence context**, enabling more precise emotion detection in continuous conversations.

---

## 🔥 Features

- **Multi-Sentence Context Understanding**  
  Uses a sliding window approach to reflect conversation history in emotion prediction.

- **Real-Time Emoji Recommendation**  
  Suggests emojis based on the predicted emotions for intuitive feedback.

- **Data Augmentation with Context Preservation**  
  Applies back-translation-based augmentation while preserving conversational context.

- **Balanced Emotion Dataset**  
  Uses a controlled dataset to address class imbalance issues.

---

<pre> ## 📁 Project Structure ``` emotion_project/ │ ├── train.py # Training pipeline with data loading, augmentation, and model training ├── predict.py # Interactive emotion prediction system (single or multi-sentence mode) ├── model.py # Model architecture and custom Dataset class ├── 한국어_연속적_대화_데이터셋.xlsx # Korean continuous dialogue dataset (multi-sentence) ├── model.pth # Trained model checkpoint (not included, shared separately) └── Multi.ipynb # Experiment notebook (optional) ``` </pre>



---

## 🚀 Quick Start

### 1. Install Requirements

```bash
pip install torch transformers pandas tqdm deep-translator
2. Download the Trained Model
The trained model file (model.pth) is not included in the repository due to size limitations.
You can download it from the following link:

🔗 Download model.pth from Google Drive

Place the downloaded model.pth file inside the emotion_project folder.

🎯 Emotion Categories
Label	Meaning
0	Fear
1	Surprise
2	Anger
3	Sadness
4	Neutral
5	Happiness
6	Disgust

💡 Notes
This project uses KLUE BERT (klue/bert-base) as the backbone.

Korean language only.

Supports both GPU and CPU.

yaml
코드 복사

---

## ✅ **핵심:**  
디렉토리 구조, 코드 블록, 테이블은 반드시  
(백틱
코드 복사




