import os
import torch
import torch.nn as nn

import argparse
import timm
import utils

import rein

import dino_variant

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--noise_rate', '-n', type=float, default=0.2)
    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['id_dataset']
    batch_size = 1
    
    if args.data == 'ham10000':
        valid_loader = utils.read_noise_datalist('./', batch_size = 1)
           
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    elif args.netsize == 'l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant

    model = rein.ReinsDinoVisionTransformer(
        **variant
    )
    model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
    model_para = torch.load(os.path.join(save_path, 'model_best.pth.tar'))
    
    model.load_state_dict(model_para['state_dict'])
    model = model.to(device)
    model.eval()
    
    total_samples = 0
    correct_count = 0        # 원래 라벨(gt)와 일치하는 경우
    noise_match_count = 0    # 노이즈 라벨(noise)와 일치하는 경우
    other_count = 0 
    
    print("노이즈 데이터에 대한 평가를 시작합니다...\n")
    with torch.no_grad():
        for batch_idx, (images, origin_labels, noise_labels) in enumerate(valid_loader):
            images = images.to(device)
            features = model.forward_features(images)
            features = features[:, 0, :]
            outputs = model.linear_rein(features)
            predicted = outputs.argmax(dim=1)
            confidence = torch.softmax(outputs, dim=-1)
            total_samples += origin_labels.size(0)
            
            correct_count += (predicted.cpu() == origin_labels).sum().item()
            noise_match_count += (predicted.cpu() == noise_labels).sum().item()
            other_count += (predicted.cpu() != origin_labels and predicted.cpu() != noise_labels).sum().item()
            
            # labels를 CPU로 이동
            # origin_labels = origin_labels.cpu()
            # noise_labels = noise_labels.cpu()
            # predicted = predicted.cpu()

            # 한 배치 내의 각 샘플에 대해 결과 출력
            for i in range(len(predicted)):
                global_index = batch_idx * batch_size + i
                print(f"샘플 {global_index}: 원래 라벨 = {origin_labels[i]}, \
                      노이즈 라벨 = {noise_labels[i]}, 예측 = {predicted[i]}, \
                      GT 신뢰도 = {confidence[i][origin_labels[i]]}, Noise 신뢰도 = {confidence[i][noise_labels[i]]}")
                
        accuracy = 100.0 * correct_count / total_samples if total_samples > 0 else 0.0
        print("\n평가 결과:")
        print(f"전체 샘플 수: {total_samples}")
        print(f"원래 라벨로 맞춘 샘플 수: {correct_count}")
        print(f"노이즈 라벨(noise)로 맞춘 샘플 수: {noise_match_count}")
        print(f"gt도 noise도 아닌 라벨로 예측한 샘플 수: {other_count}")
        print(f"정확도: {accuracy:.2f}%")

if __name__ == '__main__':
    test()