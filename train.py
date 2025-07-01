#📦 라이브러리 import (기존과 동일)
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import os
from model import AIHubEmotionDataset, EmotionClassifier, label_mapping
from tqdm import tqdm

# 🔄 증강용 추가 import
from deep_translator import GoogleTranslator
import time
import random
import torch.nn as nn
import torch.nn.functional as F

# Focal Loss 클래스 (기존과 동일)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss

#🗂️ 경로 설정 (기존과 동일)
project_path = '/content/drive/MyDrive/emotion_project'
continuous_dataset_file = f"{project_path}/한국어_연속적_대화_데이터셋.xlsx"

# 🎯 목표 감정 분포
TARGET_DISTRIBUTION = {
    '중립': 8000,    # 35%
    '분노': 3500,    # 15%
    '행복': 3000,    # 13%
    '슬픔': 2500,    # 11%
    '혐오': 2500,    # 11%
    '놀람': 2000,    # 9%
    '공포': 1500,    # 6%
}

class ContextPreservedAugmentor:
    def __init__(self):
        self.languages = ['en', 'ja', 'zh']
    
    def augment_windowed_sentence(self, windowed_sentence, lang):
        """[SEP] 토큰이 포함된 문장을 문맥 보존하며 증강"""
        try:
            # 1단계: [SEP] 토큰 제거
            clean_text = windowed_sentence.replace(" [SEP] ", " ")
            
            # 2단계: 전체 문맥 역번역
            translator_to = GoogleTranslator(source='ko', target=lang)
            intermediate = translator_to.translate(clean_text)
            time.sleep(0.02)
            
            translator_back = GoogleTranslator(source=lang, target='ko')
            back_translated = translator_back.translate(intermediate)
            time.sleep(0.02)
            
            # 3단계: 원본 [SEP] 개수 확인
            original_sep_count = windowed_sentence.count("[SEP]")
            
            if original_sep_count == 0:
                # 단일 문장
                return back_translated
            else:
                # 다중 문장: 균등 분할 후 [SEP] 재삽입
                sentences = self.split_into_sentences(back_translated, original_sep_count + 1)
                return " [SEP] ".join(sentences)
                
        except Exception as e:
            print(f"증강 실패: {e}")
            return windowed_sentence  # 실패시 원본 반환
    
    def split_into_sentences(self, text, target_count):
        """텍스트를 지정된 개수의 문장으로 분할"""
        words = text.split()
        if len(words) == 0:
            return [text] * target_count
        
        chunk_size = len(words) // target_count
        sentences = []
        
        for i in range(target_count):
            start = i * chunk_size
            if i == target_count - 1:  # 마지막 조각은 남은 모든 단어
                end = len(words)
            else:
                end = start + chunk_size
            
            sentence = " ".join(words[start:end])
            sentences.append(sentence.strip())
        
        return sentences

# ✨ 기존 load_data 함수 그대로 유지 (수정 없음)
def load_data(window_size=3):
    if not os.path.exists(continuous_dataset_file):
        raise FileNotFoundError(f"❌ 파일 없음: {continuous_dataset_file}")
    
    print(f"📂 데이터 파일 로드: {continuous_dataset_file}")
    df_raw = pd.read_excel(continuous_dataset_file, header=1)
    sentences = []
    emotions = []
    current_dialog = []
    
    for _, row in df_raw.iterrows():
        if pd.isna(row.iloc[1]):
            continue
            
        utterance = str(row.iloc[1]).strip()
        emotion = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else "중립"
        
        if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip() == 'S':
            current_dialog = []
        
        current_dialog.append(utterance)
        window = current_dialog[-window_size:] if len(current_dialog) >= window_size else current_dialog
        sentence = " [SEP] ".join(window)
        
        sentences.append(sentence)
        emotions.append(emotion)
    
    def one_hot(emo):
        vec = [0.0] * 7
        idx = label_mapping.get(str(emo).strip())
        if idx is not None:
            vec[idx] = 1.0
        return vec
    
    emotion_vecs = [one_hot(emo) for emo in emotions]
    
    df = pd.DataFrame({
        'Sentence': sentences,
        'EmotionVec': emotion_vecs,
        'Emotion': emotions
    })
    
    print(f"✅ 데이터 로드 완료: {len(df)}개 샘플 (윈도우 크기: {window_size})")
    
    emotion_dist = df['Emotion'].value_counts()
    print("\n📊 감정 분포:")
    for emotion, count in emotion_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {emotion}: {count:,}개 ({percentage:.1f}%)")
    
    return df

# ✨ 새로 추가: 전처리된 데이터 기반 증강 함수
def augment_preprocessed_data(df, target_distribution):
    """이미 슬라이딩 윈도우가 적용된 DataFrame을 기반으로 증강"""
    
    print("\n🔄 전처리된 데이터 기반 문맥 보존 증강 시작...")
    
    augmentor = ContextPreservedAugmentor()
    current_counts = df['Emotion'].value_counts()
    
    # 증강 계획 수립
    augmentation_plan = {}
    for emotion, target in target_distribution.items():
        current = current_counts.get(emotion, 0)
        if current < target and current > 0:  # 0개인 감정은 제외
            needed_multiplier = min(int(target / current), 15)  # 최대 15배 제한
            augmentation_plan[emotion] = needed_multiplier
            print(f"  📋 {emotion}: {current}개 → 목표 {target}개 (×{needed_multiplier} 증강)")
    
    if not augmentation_plan:
        print("  ⏭️ 증강이 필요한 감정이 없습니다.")
        return df
    
    # 감정별 증강 수행
    all_augmented_data = []
    
    for emotion, multiplier in augmentation_plan.items():
        print(f"\n  🎯 {emotion} 감정 증강 중... (×{multiplier})")
        
        # 해당 감정의 모든 샘플 추출
        emotion_samples = df[df['Emotion'] == emotion].copy()
        print(f"    원본 {emotion} 샘플: {len(emotion_samples)}개")
        
        # 각 언어별로 증강
        for round_num in range(multiplier):
            lang = augmentor.languages[round_num % len(augmentor.languages)]
            print(f"    라운드 {round_num+1}/{multiplier} - {lang.upper()} 증강 중...")
            
            augmented_samples = []
            
            for idx, row in emotion_samples.iterrows():
                # 문맥 보존 증강
                augmented_sentence = augmentor.augment_windowed_sentence(
                    row['Sentence'], lang
                )
                
                augmented_samples.append({
                    'Sentence': augmented_sentence,
                    'EmotionVec': row['EmotionVec'],
                    'Emotion': row['Emotion']
                })
            
            round_df = pd.DataFrame(augmented_samples)
            all_augmented_data.append(round_df)
            
            print(f"      완료: {len(augmented_samples)}개 샘플 증강")
    
    # 원본과 증강 데이터 합치기
    if all_augmented_data:
        augmented_df = pd.concat(all_augmented_data, ignore_index=True)
        combined_df = pd.concat([df, augmented_df], ignore_index=True)
        
        print(f"\n📊 증강 후 총 데이터: {len(combined_df)}개")
        
        # 증강 후 감정 분포
        final_counts = combined_df['Emotion'].value_counts()
        print("증강 후 감정 분포:")
        for emotion, count in final_counts.items():
            percentage = (count / len(combined_df)) * 100
            print(f"  {emotion}: {count:,}개 ({percentage:.1f}%)")
        
        return combined_df
    else:
        return df

def balance_dataset(df, target_distribution):
    """목표 분포에 맞춰 균형 조정"""
    
    print("\n⚖️ 균형 샘플링 적용...")
    
    balanced_parts = []
    current_counts = df['Emotion'].value_counts()
    
    for emotion, target_count in target_distribution.items():
        emotion_data = df[df['Emotion'] == emotion]
        current_count = len(emotion_data)
        
        if current_count >= target_count:
            sampled = emotion_data.sample(n=target_count, random_state=42)
            print(f"    {emotion}: {current_count:,} → {target_count:,} (샘플링)")
        else:
            sampled = emotion_data
            print(f"    {emotion}: {current_count:,} → {current_count:,} (전부 사용)")
        
        balanced_parts.append(sampled)
    
    balanced_df = pd.concat(balanced_parts, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"📊 최종 균형 데이터: {len(balanced_df)}개")
    return balanced_df

# ✨ 수정된 train 함수 (load_data 후 증강 추가)
def train(start_epoch=0, total_epochs=10, window_size=3):
    pretrained_model_name = "klue/bert-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 사용 디바이스: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    # 1. 기존 방식으로 데이터 로드 (슬라이딩 윈도우 적용됨)
    df = load_data(window_size)
    
    # 2. 전처리된 데이터를 기반으로 문맥 보존 증강
    augmented_df = augment_preprocessed_data(df, TARGET_DISTRIBUTION)
    
    # 3. 균형 샘플링
    final_df = balance_dataset(augmented_df, TARGET_DISTRIBUTION)
    
    print("\n📊 데이터 분할 (학습 80% : 평가 20%)")
    train_df = final_df.sample(frac=0.8, random_state=42)
    eval_df = final_df.drop(train_df.index)
    
    print(f"✅ 학습 데이터: {len(train_df):,}개 ({len(train_df)/len(final_df)*100:.1f}%)")
    print(f"✅ 평가 데이터: {len(eval_df):,}개 ({len(eval_df)/len(final_df)*100:.1f}%)")
    
    # 데이터셋 및 데이터로더 생성 (기존과 동일)
    train_dataset = AIHubEmotionDataset(train_df, tokenizer)
    eval_dataset = AIHubEmotionDataset(eval_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32)

    # 모델 생성 (기존과 동일)
    model = EmotionClassifier(pretrained_model_name, num_classes=7).to(device)

    # 옵티마이저 및 손실함수 설정
    optimizer = AdamW(model.parameters(), lr=3e-5)
    loss_fn = FocalLoss(alpha=1.0, gamma=2.0)  # Focal Loss 사용
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=len(train_loader) * 2,
        num_training_steps=len(train_loader)*total_epochs
    )

    print(f"\n🚀 Focal Loss로 학습 시작 (총 {total_epochs}개 에포크)")
    
    best_eval_loss = float('inf')
    
    # 학습 루프 (기존과 완전 동일)
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_train_loss = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} - Training")
        for batch in train_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_eval_loss = 0
        
        eval_loop = tqdm(eval_loader, desc=f"Epoch {epoch+1}/{total_epochs} - Evaluation")
        with torch.no_grad():
            for batch in eval_loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                
                total_eval_loss += loss.item()
                eval_loop.set_postfix(loss=loss.item())
        
        avg_eval_loss = total_eval_loss / len(eval_loader)
        
        print(f"✅ Epoch {epoch+1}")
        print(f"   📚 Train Loss: {avg_train_loss:.4f}")
        print(f"   📊 Eval Loss:  {avg_eval_loss:.4f}")
        
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            best_model_path = f"{project_path}/best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"   🏆 최고 성능 모델 저장! (Eval Loss: {avg_eval_loss:.4f})")

    final_save_path = f"{project_path}/model.pth"
    torch.save(model.state_dict(), final_save_path)
    print(f"\n✅ 학습 완료!")
    print(f"📁 최종 모델: {final_save_path}")
    print(f"🏆 최고 성능 모델: {best_model_path}")

# 실행
if __name__ == "__main__":
    train(total_epochs=8)