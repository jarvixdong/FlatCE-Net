import torch


class BasicTrainer:
    def __init__(self, model, optimizer, scheduler, config, device, logger):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = scheduler  # 使用统一名称
        self.config = config
        self.device = device
        self.logger = logger
        
        # 初始化必要属性
        self.step = 0
        self.start_epoch = 1
        self.end_epoch = self.config['epoch_num']

    def train(self):
        for epoch in range(self.start_epoch,self.end_epoch):
            self._train_epoch(epoch)
            
            
    def _save_checkpoint(self, epoch, save_path):
        if hasattr(self.model, "module"):
            model = self.model.module
        else:
            model = self.model

        if hasattr(model, "saved_model"):
            model = model.saved_model()

        to_save = {
            'epoch': epoch,
            'step': self.step,
            'model': model.state_dict(),
            'optim': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'config': self.config
        }
        torch.save(to_save, save_path)
        torch.cuda.empty_cache()

    def _resume_checkpoint(self, resume_path, is_finetune):
        # 假设 load_model 返回 (model, checkpoint)
        model, checkpoint = self.model.load_model(resume_path, device=self.device)
        if not is_finetune:
            self.step = checkpoint["step"]
            self.start_epoch = checkpoint['epoch'] + 1
            self.optimizer.load_state_dict(checkpoint['optim'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.logger.info('resume params from {}, epoch={}, step={}'.format(
            resume_path, self.start_epoch, self.step)
        )
        return model

class Trainer(BasicTrainer):
    def __init__(self, model, optimizer, scheduler, train_loader, valid_loader, config, device, logger):
        # 确保传递所有父类需要的参数
        super().__init__(model, optimizer, scheduler, config, device, logger)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        

    def _train_epoch(self,epoch):
        
        for idx, batch in enumerate(self.train_loader):
            batch = {'inputs': batch[0].to(self.device),
                    'targets': batch[1].to(self.device),
                    'Vpinv': batch[2].to(self.device)}
            
            self.optimizer.zero_grad()
            
            output, loss = self.model(batch)
            target = batch['targets']
            # loss = criterion(output, target)

            with torch.no_grad():
                # nmse_train = cal_NMSE4(output.detach().cpu().numpy(),target.detach().cpu().numpy())
                diff_model = (output - target) ** 2
                target_sq = target ** 2
                sum_mse    += diff_model.sum().item()
                sum_target += target_sq.sum().item()

            loss.backward()
            self.optimizer.step()

        train_avg_nmse = sum_mse / sum_target
    
        # break
        nmmse,ls_nmse = valid(model,dataloader)
        # scheduler.step(nmmse) 
        self.scheduler.step()      
        
        nmmse = float(f"{nmmse:.6f}")  # 保留 6 位小数
        ls_nmse = float(f"{ls_nmse:.6f}")  # 保留 6 位小数
        nmse_train = float(f"{train_avg_nmse:.6f}")  # 保留 6 位小数 
    
    def __valid_epoch(self,epoch):
        for idx, batch in enumerate(self.valid_loader):
            with torch.no_grad():
                # 将输入/目标/等辅助参数放到 GPU（可视实际需求而定）
                batch = {'inputs': batch[0].to(self.device),
                    'targets': batch[1].to(self.device),
                    'Vpinv': batch[2].to(self.device)}
                # vpinv  = batch[2].to(device)  # 如确有需要可以继续用

                # 模型前向推理
                outputs, loss = self.model(batch)

                # # 直接在 GPU 上计算误差平方
                # diff_ls    = (inputs - targets) ** 2
                # diff_model = (outputs - targets) ** 2
                # targets_sq = targets ** 2

                # # 将每个batch的误差、真值平方和累加到Python标量中
                # sum_ls_mse += diff_ls.sum().item()
                # sum_mse    += diff_model.sum().item()
                # sum_target += targets_sq.sum().item()

        # # 计算平均 NMSE 和 LS NMSE（全局归一化）
        # avg_nmse = sum_mse / sum_target
        # avg_ls_nmse = sum_ls_mse / sum_target
        
        return loss
        







def valid(model, dataloader):
    model.eval()

    sum_ls_mse = 0.0   # 累计 LS 误差平方和
    sum_mse = 0.0      # 累计网络输出误差平方和
    sum_target = 0.0   # 累计真值平方和

    with torch.no_grad():
        for idx, batch in enumerate(dataloader.valid_loader):
            # 将输入/目标/等辅助参数放到 GPU（可视实际需求而定）
            inputs = batch[0].cuda(non_blocking=True)
            targets = batch[1].cuda(non_blocking=True)
            # vpinv  = batch[2].to(device)  # 如确有需要可以继续用

            # 模型前向推理
            outputs = model(inputs)

            # 直接在 GPU 上计算误差平方
            diff_ls    = (inputs - targets) ** 2
            diff_model = (outputs - targets) ** 2
            targets_sq = targets ** 2

            # 将每个batch的误差、真值平方和累加到Python标量中
            sum_ls_mse += diff_ls.sum().item()
            sum_mse    += diff_model.sum().item()
            sum_target += targets_sq.sum().item()

    # 计算平均 NMSE 和 LS NMSE（全局归一化）
    avg_nmse = sum_mse / sum_target
    avg_ls_nmse = sum_ls_mse / sum_target
    
    return avg_nmse, avg_ls_nmse


    
def train(model, criterion, optimizer, scheduler ,dataloader, save_path, epochs=10):
    
    # criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    print('loss function::',criterion)
    # rayleigh_loss = RayleighKLLoss()
    rayleigh_loss = RayleighKLLoss_mat2()
    # criterion = nn.SmoothL1Loss(beta=1.0)
    best_nmmse = 100
    for epoch in range(epochs):
        model.train()
        sum_mse = 0.0      # 累计网络输出误差平方和
        sum_target = 0.0   # 累计真值平方和
        for idx, batch in enumerate(dataloader.train_loader):
            # batch = {'inputs': batch[0].to(device),
            #         'targets': batch[1].to(device),
            #         'Vpinv': batch[2].to(device)}
            batch = {'inputs': batch[0].cuda(non_blocking=True),
                    'targets': batch[1].cuda(non_blocking=True)}
            
            inputdata = batch['inputs']
            # print('input data:',inputdata[0][0])
            target = batch['targets']
            # vpinv = batch['Vpinv']

            output = model(inputdata)
            loss = criterion(output, target)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # nmse_train = cal_NMSE4(output.detach().cpu().numpy(),target.detach().cpu().numpy())
                diff_model = (output - target) ** 2
                target_sq = target ** 2
                sum_mse    += diff_model.sum().item()
                sum_target += target_sq.sum().item()
        
        train_avg_nmse = sum_mse / sum_target
    
        # break
        nmmse,ls_nmse = valid(model,dataloader)
        # scheduler.step(nmmse) 
        scheduler.step()      
        
        nmmse = float(f"{nmmse:.6f}")  # 保留 6 位小数
        ls_nmse = float(f"{ls_nmse:.6f}")  # 保留 6 位小数
        nmse_train = float(f"{train_avg_nmse:.6f}")  # 保留 6 位小数

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, Train_MMSE: {nmse_train}, NMMSE: {nmmse}, LS_NMSE: {ls_nmse}, Lr: {optimizer.param_groups[0]['lr']}")

        # Save model checkpoint for each epoch
        model_save_path = os.path.join(save_path, "save_models/",f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        # print(f"Model saved at {model_save_path}")

        # Save best model based on validation loss
        if nmmse < best_nmmse:
            best_nmmse = nmmse
            best_model_path = os.path.join(save_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            # print(f"Best model updated and saved at {best_model_path}")
            