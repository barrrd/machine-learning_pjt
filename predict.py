import torch
from transformers import AutoTokenizer
from model import EmotionClassifier

project_path = '/content/drive/MyDrive/emotion_project'

reverse_label_mapping = {
    0: "공포", 1: "놀람", 2: "분노",
    3: "슬픔", 4: "중립", 5: "행복", 6: "혐오"
}

emoji_dict = {
    0: ["😱", "😨", "👻"],       # 공포
    1: ["😲", "😯", "😳"],       # 놀람
    2: ["😡", "😠", "🤬"],       # 분노
    3: ["😭", "😢", "😞"],       # 슬픔
    4: ["😐", "😑", "🙂"],       # 중립
    5: ["😄", "😂", "😝"],       # 행복
    6: ["🤢", "🤮", "😖"]        # 혐오
}

def predict_conversation(window_size=3):
    """슬라이딩 윈도우 대화 예측"""
    pretrained_model_name = "klue/bert-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    model = EmotionClassifier(pretrained_model_name, num_classes=7).to(device)
    model.load_state_dict(torch.load(f"{project_path}/model.pth", map_location=device))
    model.eval()

    print(f"🗣️ 대화 시작! (윈도우 크기: {window_size}, 종료: exit)")
    conversation = []

    while True:
        user_input = input("💬 입력: ").strip()
        if user_input.lower() == "exit":
            break
        
        conversation.append(user_input)
        
        # 슬라이딩 윈도우로 입력 구성
        window = conversation[-window_size:] if len(conversation) >= window_size else conversation
        
        if len(window) == 1:
            full_text = window[0]
        else:
            full_text = " [SEP] ".join(window)
            print(f"📖 문맥: {' → '.join(window[:-1])}")
        
        # 예측
        inputs = tokenizer(full_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

        # 상위 2개 감정
        indexed = list(enumerate(probs))
        top2 = sorted(indexed, key=lambda x: x[1], reverse=True)[:2]

        first_idx, first_score = top2[0]
        second_idx, second_score = top2[1]
        
        label1 = reverse_label_mapping[first_idx]
        label2 = reverse_label_mapping[second_idx]
        
        print(f"🎯 감정: {label1} ({first_score:.3f}), {label2} ({second_score:.3f})")

        # 이모지 추천
        emoji_list1 = emoji_dict.get(first_idx, ["❓"])
        emoji_list2 = emoji_dict.get(second_idx, ["❓"])
        
        if second_score >= 0.5:
            emojis = emoji_list1[:2] + [emoji_list2[0]]
        else:
            emojis = emoji_list1[:3]

        print(f"🎨 추천 이모지: {' '.join(emojis)}\n")

def predict_single():
    """단일 문장 예측 (기존 방식)"""
    pretrained_model_name = "klue/bert-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    model = EmotionClassifier(pretrained_model_name, num_classes=7).to(device)
    model.load_state_dict(torch.load(f"{project_path}/model.pth", map_location=device))
    model.eval()

    while True:
        text = input("문장 입력 (종료: exit): ").strip()
        if text.lower() == "exit":
            break
        
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

        indexed = list(enumerate(probs))
        top2 = sorted(indexed, key=lambda x: x[1], reverse=True)[:2]

        first_idx, first_score = top2[0]
        second_idx, second_score = top2[1]

        label1 = reverse_label_mapping[first_idx]
        label2 = reverse_label_mapping[second_idx]
        emoji_list1 = emoji_dict.get(first_idx, ["❓"])
        emoji_list2 = emoji_dict.get(second_idx, ["❓"])

        print(f"감정: {label1} ({first_score:.2f}), {label2} ({second_score:.2f})")

        if second_score >= 0.5:
            emojis = emoji_list1[:2] + [emoji_list2[0]]
        else:
            emojis = emoji_list1[:3]

        print(f"이모지: {' '.join(emojis)}\n")

if __name__ == "__main__":
    print("모드 선택:")
    print("1. 대화 모드 (슬라이딩 윈도우)")
    print("2. 단일 문장 모드")
    
    choice = input("선택 (1/2): ").strip()
    
    if choice == "1":
        predict_conversation()
    else:
        predict_single()