import torch
import torch.nn as nn
import torch.optim as optim

def training_classifier(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    # 返回参数的个数，总的和训练的
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nStart training, parameters total: {}, trainable: {}\n'.format(total, trainable))
    model.train() # 将 model 的模式改为 train, 使得 optimizer 可以更新 model 的参数
    criterion = nn.BCELoss() # 损失函数定义为 cross entropy loss
    t_batch = len(train)
    v_batch = len(valid)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_loss, best_loss = 0, 1e5
    for epoch in range(n_epoch):
        total_loss = 0
        # training
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.float) # device 为 cuda, 将 inputs 转为 torch.cuda.LongTensor
            labels = labels.to(device, dtype=torch.float) # device 为 cuda, 将 labels 转为 torch.cuda.FloatTrnsor, float 为了criterion计算
            optimizer.zero_grad()
            #print("epoch：", epoch, "的第" , i, "个inputs", inputs.data.size(), "labels", labels.data.size())
            inputs=inputs.permute(0,2,1) # For sei model
            outputs = model(inputs)
            outputs = outputs.squeeze() # 去掉外层的 dimension, 使得 outputs 可以被 criterion()
            loss = criterion(outputs, labels) # 计算 model 的 training loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item()), end='\r')
        print('\nTrain | Loss:{:.5f}'.format(total_loss/t_batch))
        train_loss = total_loss/t_batch
        
        # validation
        model.eval() # 将 model 转为 eval 模式, 固定住 model 的参数
        with torch.no_grad():
            total_loss = 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                inputs=inputs.permute(0,2,1) # For sei model
                outputs = model(inputs)
                #outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} ".format(total_loss/v_batch))
            
            if total_loss < best_loss:
                # 如果 validation 的结果优于之前的结果, 就保存当前模型
                best_loss = total_loss
                torch.save(model, model_dir)
                print('saving model with loss {:.3f}'.format(total_loss))
                
                
        print('-----------------------------------------------')
        model.train() # 将 model 重新调整为train, 进行后续训练