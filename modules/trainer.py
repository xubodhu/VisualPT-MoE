import torch
from torch import optim
from tqdm import tqdm
import random
from sklearn.metrics import classification_report as sk_classification_report
from seqeval.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup
from ner_evaluate import evaluate
import time

from .metrics import eval_result

class BaseTrainer(object):
    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()



class NERTrainer(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None, label_map=None,
                 args=None, logger=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        self.logger = logger
        self.label_map = label_map
        self.refresh_step = 2
        self.best_evaluate_f1 = 0
        self.best_dev_metric = 0
        self.best_train_metric = 0
        self.best_dev_epoch = None
        self.best_train_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args

    def train(self):
        self.multiModal_before_train()

        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))


        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0

            for epoch in range(1, self.args.num_epochs + 1):
                y_true, y_pred = [], []
                y_true_idx, y_pred_idx = [], []
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                begin = time.time()
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    attention_mask, labels, logits, loss = self._step(batch)
                    avg_loss += loss.detach().cpu().item()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if isinstance(logits, torch.Tensor):  # CRF return lists
                        logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                    label_ids = labels.to('cpu').numpy()
                    input_mask = attention_mask.to('cpu').numpy()
                    label_map = {idx: label for label, idx in self.label_map.items()}
                    reverse_label_map = {label: idx for label, idx in self.label_map.items()}

                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_label_idx = []
                        true_predict = []
                        true_predict_idx = []
                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_ids[row][column] != -100:
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_label_idx.append(label_ids[row][column])
                                    true_predict.append(label_map[logits[row][column]])
                                    true_predict_idx.append(logits[row][column])
                            else:
                                break
                        y_true.append(true_label)
                        y_true_idx.append(true_label_idx)
                        y_pred.append(true_predict)
                        y_pred_idx.append(true_predict_idx)

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)

                        avg_loss = 0
                end = time.time()
                self.logger.info("train time = %s", str(end -begin))
                results = classification_report(y_true, y_pred, digits=4)
                acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, y_pred, reverse_label_map)

                self.logger.info("***** Train Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[0].split('    ')[3])

                self.logger.info("Epoch {}/{}, best train f1: {}, best epoch: {}, current train f1 score: {}." \
                                 .format(epoch, self.args.num_epochs, self.best_train_metric, self.best_train_epoch,
                                         f1_score))
                self.logger.info("The Evaluate F1 = %s", str(f1))

                if f1_score > self.best_train_metric:
                    self.best_train_metric = f1_score
                    self.best_train_epoch = epoch

                
                self.evaluate(epoch)  # generator to dev.

            torch.cuda.empty_cache()

            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch,
                                                                                                    self.best_dev_metric))
            self.logger.info("The max_f1 = %s", str(self.best_dev_metric))
            self.logger.info("The max_evaluate_f1 = %s", str(self.best_evaluate_f1))
            self.test()


    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        y_true, y_pred = [], []
        y_true_idx, y_pred_idx = [], []
        step = 0
        begin = time.time()
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                             batch)  # to cpu/cuda device
                    attention_mask, labels, logits, loss = self._step(batch)  # logits: batch, seq, num_labels
                    total_loss += loss.detach().cpu().item()

                    if isinstance(logits, torch.Tensor):
                        logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                    label_ids = labels.detach().cpu().numpy()
                    input_mask = attention_mask.detach().cpu().numpy()
                    label_map = {idx: label for label, idx in self.label_map.items()}
                    reverse_label_map = {label: idx for label, idx in self.label_map.items()}
                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_label_idx = []
                        true_predict = []
                        true_predict_idx = []
                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_ids[row][column] != -100:
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_label_idx.append(label_ids[row][column])
                                    true_predict.append(label_map[logits[row][column]])
                                    true_predict_idx.append(logits[row][column])
                            else:
                                break
                        y_true.append(true_label)
                        y_true_idx.append(true_label_idx)
                        y_pred.append(true_predict)
                        y_pred_idx.append(true_predict_idx)

                    pbar.update()
                pbar.close()
                results = classification_report(y_true, y_pred, digits=4)
                acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, y_pred, reverse_label_map)
                self.logger.info("***** Dev Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1])


                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}." \
                                 .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch,
                                         f1_score))
                self.logger.info("The Evaluate F1 = %s", str(f1))
                if f1_score >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = f1_score  # update best metric(f1 score)
                if f1 >= self.best_evaluate_f1:
                    self.best_evaluate_f1 = f1
                    torch.save(self.model.state_dict(), "./weights/best_model.pth")
                    self.logger.info("Save best model at {}".format("./weights/best_model.pth"))
        end = time.time()
        self.logger.info("test time = %s", str(end - begin))
        self.model.train()



    def test(self):
        self.model.eval()
        self.model.to(self.args.device)
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)


        self.logger.info("Loading model from {}".format("./weights/best_model.pth"))
        self.model.load_state_dict(torch.load("./weights/best_model.pth"))
        self.logger.info("Load model successful!")
        y_true, y_pred = [], []
        y_true_idx, y_pred_idx = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                             batch)  # to cpu/cuda device
                    attention_mask, labels, logits, loss = self._step(batch)  # logits: batch, seq, num_labels
                    total_loss += loss.detach().cpu().item()

                    if isinstance(logits, torch.Tensor):
                        logits = logits.argmax(-1).detach().cpu().tolist()  # batch, seq, 1
                    label_ids = labels.detach().cpu().numpy()
                    input_mask = attention_mask.detach().cpu().numpy()
                    label_map = {idx: label for label, idx in self.label_map.items()}
                    reverse_label_map = {label: idx for label, idx in self.label_map.items()}
                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_label_idx = []
                        true_predict = []
                        true_predict_idx = []
                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_ids[row][column] != -100:
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_label_idx.append(label_ids[row][column])
                                    true_predict.append(label_map[logits[row][column]])
                                    true_predict_idx.append(logits[row][column])
                            else:
                                break
                        y_true.append(true_label)
                        y_true_idx.append(true_label_idx)
                        y_pred.append(true_predict)
                        y_pred_idx.append(true_predict_idx)
                    pbar.update()
                # evaluate done
                pbar.close()

                results = classification_report(y_true, y_pred, digits=4)
                acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, y_pred, reverse_label_map)

                self.logger.info("***** Test Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1])

                total_loss = 0
                self.logger.info("Test f1 score: {}.".format(f1_score))
                self.logger.info("The Evaluate F1 = %s", str(f1))

        self.model.train()

    def _step(self, batch):
        input_ids, token_type_ids, attention_mask, labels, main_features, aux_features, \
        tag_embeddings, caption_embeddings = batch

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                labels=labels, main_features=main_features, aux_features=aux_features, tag_embeddings=tag_embeddings,caption_embeddings=caption_embeddings)
        logits, loss = output.logits, output.loss
        return attention_mask, labels, logits, loss

    def bert_before_train(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)

        self.model.to(self.args.device)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)

    def multiModal_before_train(self):
        # bert lr
        parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'bert' in name:
                params['params'].append(param)
        parameters.append(params)

        # prompt lr
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'encoder_conv' in name or 'gates' in name:
                params['params'].append(param)
        parameters.append(params)

        params = {'lr':self.args.final_lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if name.startswith('fc') or name.startswith('aux'):
                params['params'].append(param)
        parameters.append(params)


        self.optimizer = optim.AdamW(parameters)


        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)


class RETrainer(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None, args=None,
                 logger=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        self.logger = logger
        self.re_dict = processor.get_relation_dict()
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_dev_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args


    def train(self):
        self.before_multimodal_train()


        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))



        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs + 1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))

                true_labels, pred_labels = [], []

                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    logits, loss, labels = self._step(batch)
                    avg_loss += loss.detach().cpu().item()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()


                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())


                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)

                        avg_loss = 0

                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels,
                                                     labels=list(self.re_dict.values())[1:],
                                                     target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("------------  This is the Training Result ------------")
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)
                self.logger.info("The Training f1 = %s" % str(micro_f1))

                self.evaluate(epoch)  # generator to dev.

            pbar.close()
            self.pbar = None
           
            self.logger.info(
                "Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch,
                                                                                         self.best_dev_metric))

            self.test()
    
    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        step = 0
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                             batch)  # to cpu/cuda device
                    logits, loss, labels = self._step(batch)
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels,
                                                     labels=list(self.re_dict.values())[1:],
                                                     target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, acc: {}." \
                                 .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch,
                                         micro_f1, acc))
                if micro_f1 >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = micro_f1  # update best metric(f1 score)
                    torch.save(self.model.state_dict(), "./weights/best_model_mre.pth")
                    self.logger.info("Save best model at {}".format("./weights/best_model_mre.pth"))

        self.model.train()

    def test(self):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("Loading model from {}".format("./weights/best_model_mre.pth"))
        self.model.load_state_dict(torch.load("./weights/best_model_mre.pth"))
        self.logger.info("Load model successful!")
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                             batch)  # to cpu/cuda device
                    loss, logits, labels = self._step(batch)  # logits: batch, 3
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())

                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels,
                                                     labels=list(self.re_dict.values())[1:],
                                                     target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc'] * 100, 4), round(result['micro_f1'] * 100, 4)
                if self.writer:
                    self.writer.add_scalar(tag='test_acc', scalar_value=acc)  # tensorbordx
                    self.writer.add_scalar(tag='test_f1', scalar_value=micro_f1)  # tensorbordx
                    self.writer.add_scalar(tag='test_loss',
                                           scalar_value=total_loss / len(self.test_data))  # tensorbordx
                total_loss = 0
                self.logger.info("Test f1 score: {}, acc: {}.".format(micro_f1, acc))

        self.model.train()

    def _step(self, batch):

        input_ids, token_type_ids, attention_mask, labels, resnet_features_main, resnet_features_aux, \
        tag_embeddings, caption_embeddings = batch
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                labels=labels, resnet_features_main=resnet_features_main, resnet_features_aux=resnet_features_aux, tag_embeddings=tag_embeddings,
                                caption_embeddings=caption_embeddings)

        logits, loss = output.logits, output.loss

        return logits, loss, labels


    def before_train(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 1e-2},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)

    def before_multimodal_train(self):
        # bert lr
        parameters = []
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'bert' in name:
                params['params'].append(param)
        parameters.append(params)

        # prompt lr
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'encoder_conv' in name or 'gates' in name:
                params['params'].append(param)
        parameters.append(params)

        params = {'lr': self.args.final_lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'fc' in name:
                params['params'].append(param)
        parameters.append(params)



        self.optimizer = optim.AdamW(parameters, lr=self.args.lr)

        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)
