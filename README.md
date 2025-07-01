🗣️ Korean Multi-Sentence Based Emoji Recommendation and Emotion Recognition System
This project is a Korean emotion recognition system that predicts the user's emotion based on multi-sentence conversational context and recommends suitable emojis accordingly.
The system considers both single-sentence and sliding window-based multi-sentence context, enabling more precise emotion detection in continuous conversations.

🔥 Features

Multi-Sentence Context Understanding
Uses a sliding window approach to reflect conversation history in emotion prediction.
Real-Time Emoji Recommendation
Suggests emojis based on the predicted emotions for intuitive feedback.
Data Augmentation with Context Preservation
Applies back-translation-based augmentation while preserving conversational context.
Balanced Emotion Dataset
Uses a controlled dataset to address class imbalance issues.


📁 Project Structure
emotion_project/
│
├── train.py                              # Training pipeline with data 
├── predict.py                            # Interactive emotion 
├── model.py                              # Model architecture and custom 
├── 한국어_연속적_대화_데이터셋.xlsx           # Korean continuous 
├── model.pth                             # Trained model checkpoint (not 
└── Multi.ipynb                           # Experiment notebook (optional)

🚀 Quick Start
1. Install Requirements
bashpip install torch transformers pandas tqdm deep-translator
2. Download the Trained Model
The trained model file (model.pth) is not included in the repository due to size limitations.
You can download it from the following link:
🔗 Download model.pth from Google Drive
Place the downloaded model.pth file inside the emotion_project folder.
3. Run Prediction
bashpython predict.py

🎯 Emotion Categories
LabelMeaning0Fear1Surprise2Anger3Sadness4Neutral5Happiness6Disgust

🔧 Usage Examples

Multi-Sentence Mode
pythonpython predict.py --mode multi
# Input: ["어제 시험을 봤는데", "결과가 별로 안 좋았어", "정말 속상해"]
# Output: Emotion: Sadness (3), Emoji: 😢
