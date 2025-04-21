import os
import torch
import torch.nn as nn

import argparse
import timm
import utils

import rein

import dino_variant


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--netsize', default='s', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--noise_rate', '-n', type=float, default=0.2)
    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.data+'_h100.json')
    device = 'cuda:'+args.gpu
    save_path = os.path.join(config['save_path'], args.save_path)
    data_path = config['id_dataset']
    batch_size = int(config['batch_size'])
    init_epoch = int(config['init_epoch'])
    max_epoch = int(config['epoch'])
    noise_rate = args.noise_rate

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    lr_decay = [int(0.5*max_epoch), int(0.75*max_epoch), int(0.9*max_epoch)]

    if args.data == 'ham10000':
        train_loader, valid_loader = utils.get_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)
    elif args.data == 'aptos':
        train_loader, valid_loader = utils.get_aptos_noise_dataset(data_path, noise_rate=noise_rate, batch_size = batch_size)
    elif 'mnist' in args.data:
        train_loader, valid_loader = utils.get_mnist_noise_dataset(args.data, noise_rate=noise_rate, batch_size = batch_size)
    elif 'cifar' in args.data:
        train_loader, valid_loader = utils.get_cifar_noise_dataset(args.data, data_path, batch_size = batch_size,  noise_rate=noise_rate)
               
    if args.netsize == 's':
        model_load = dino_variant._small_dino
        variant = dino_variant._small_variant
    elif args.netsize == 'b':
        model_load = dino_variant._base_dino
        variant = dino_variant._base_variant
    elif args.netsize == 'l':
        model_load = dino_variant._large_dino
        variant = dino_variant._large_variant
    # model = timm.create_model(network, pretrained=True, num_classes=2) 
    model = torch.hub.load('facebookresearch/dinov2', model_load)
    dino_state_dict = model.state_dict()

    model = rein.DualReinsDinoVisionTransformer(
        **variant
    )
    model.load_state_dict(dino_state_dict, strict=False)
    model.linear = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.linear_rein = nn.Linear(variant['embed_dim'], config['num_classes'])
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 1e-5)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 1) 
    print(train_loader.dataset[0][0].shape)

    print('## Trainable parameters')
    model.eval()
    model.train1()
    for n, p in model.named_parameters():
        if p.requires_grad == True:
            print(n)
    model.eval()
    model.train2()
    for n, p in model.named_parameters():
        if p.requires_grad == True:
            print(n)
    
    ## init training 1
    print("-init 1-")
    for epoch in range(init_epoch):
        correct = 0
        correct_linear = 0
        total = 0
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            model.train1()
            
            features_rein = model.forward_features1(inputs)
            features_rein = features_rein[:, 0, :]
            outputs = model.linear_rein(features_rein)
            
            with torch.no_grad():
                features_ = model.forward_features_no_rein(inputs)
                features_ = features_[:, 0, :]
            outputs_ = model.linear(features_)

            with torch.no_grad():
                pred = (outputs_).max(1).indices
                linear_accurate = (pred==targets)
                
            loss_rein = linear_accurate*criterion(outputs, targets)
            loss_linear = criterion(outputs_, targets)
            loss = loss_linear.mean()+loss_rein.mean()
            loss.backward()
            optimizer.step()
            
            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()
            
            _, predicted = outputs_[:len(targets)].max(1)            
            correct_linear += predicted.eq(targets).sum().item()
            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc1: %.3f%% | LinearAcc: %.3f%% | (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, 100.*correct_linear/total, correct, total), end = '')
        # train_accuracy = correct/total
        # train_avg_loss = total_loss/len(train_loader)
        print()
    
    ## init training 2
    print("-init 2-")
    for epoch in range(init_epoch):
        correct = 0
        correct_linear = 0
        total = 0
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            model.train2()
            
            features_rein = model.forward_features2(inputs)
            features_rein = features_rein[:, 0, :]
            outputs = model.linear_rein(features_rein)
            
            with torch.no_grad():
                features_ = model.forward_features_no_rein(inputs)
                features_ = features_[:, 0, :]
            outputs_ = model.linear(features_)

            with torch.no_grad():
                pred = (outputs_).max(1).indices
                linear_accurate = (pred==targets)
                
            loss_rein = linear_accurate*criterion(outputs, targets)
            loss_linear = criterion(outputs_, targets)
            loss = loss_linear.mean()+loss_rein.mean()
            loss.backward()
            optimizer.step()
            
            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()
            
            _, predicted = outputs_[:len(targets)].max(1)            
            correct_linear += predicted.eq(targets).sum().item()
            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc1: %.3f%% | LinearAcc: %.3f%% | (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, 100.*correct_linear/total, correct, total), end = '')
        print()
    

    # co teaching
    print('-co teaching-')
    avg_accuracy = 0.0
    for epoch in range(max_epoch):
        ## training
        total_loss = 0
        total = 0
        correct1 = 0
        correct2 = 0
        correct_dual = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            inputs1, targets1 = inputs[:len(inputs)//2], targets[:len(inputs)//2]
            inputs2, targets2 = inputs[len(inputs)//2:], targets[len(inputs)//2:]
            
            model.train_all()
            
            features_rein1 = model.forward_features1(inputs1)
            features_rein1 = features_rein1[:, 0, :]
            outputs1 = model.linear_rein(features_rein1)
            
            features_rein2 = model.forward_features2(inputs2)
            features_rein2 = features_rein2[:, 0, :]
            outputs2 = model.linear_rein(features_rein2)
            
            with torch.no_grad():
                features_dual_rein = model.forward_dual_features(inputs)
                features_dual_rein = features_dual_rein[:, 0, :]
            outputs_dual = model.linear_rein(features_dual_rein)
            # print(outputs.shape, outputs_.shape)

            with torch.no_grad():
                pred1 = (outputs1).max(1).indices
                pred2 = (outputs2).max(1).indices
                
                linear_accurate1 = (pred1==targets1)
                linear_accurate2 = (pred2==targets2)
                linear_dual = torch.cat([linear_accurate2, linear_accurate1], dim=0)

            loss_rein1 = linear_accurate2*criterion(outputs1, targets1)
            loss_rein2 = linear_accurate1*criterion(outputs2, targets2)
            loss_dual = linear_dual*criterion(outputs_dual, targets)
            
            loss = loss_rein1.mean() + loss_rein2.mean() + loss_dual.mean()
            loss.backward()            
            optimizer.step() # + outputs_

            total_loss += loss
            total += targets.size(0)
            _, predicted1 = outputs1[:len(targets1)].max(1)            
            correct1 += predicted1.eq(targets1).sum().item()
            
            _, predicted2 = outputs2[:len(targets2)].max(1)            
            correct2 += predicted2.eq(targets2).sum().item()   
            
            _, predicted_dual = outputs_dual[:len(targets)].max(1)            
            correct_dual += predicted_dual.eq(targets).sum().item()   

            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc1: %.3f%% |  Acc2: %.3f%% |  Acc dual: %.3f%% | (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct1/total/2, 100.*correct2/total/2, 100.*correct_dual/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy_ = utils.validation_accuracy(model, valid_loader, device, mode='dual')
        scheduler.step()
        if epoch >= max_epoch-10:
            avg_accuracy += valid_accuracy_ 
        saver.save_checkpoint(epoch, metric = valid_accuracy_)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID_1 [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy_))
        print(scheduler.get_last_lr())
    with open(os.path.join(save_path, 'avgacc.txt'), 'w') as f:
        f.write(str(avg_accuracy/10))
if __name__ =='__main__':
    train()