import torch
from torch.cuda.amp import autocast
from utils.misc import nms

def train_step(images, true_boxes, true_classes, model, optim, amp, scaler,
               metrics, device):
    images = images.to(device)
    true_boxes = [x.to(device) for x in true_boxes]
    true_classes = [x.to(device) for x in true_classes]

    optim.zero_grad()
    with autocast(enabled=amp):
        preds = model(images)
        regression_preds, class_preds = preds
        loss, box_loss, cls_loss = model.compute_loss(preds, true_boxes, true_classes)
        print(f'loss after training  : {loss}')
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    loss = loss.item()
    metrics['loss'].update(loss, images.shape[0])
    
    box_loss = box_loss.cpu().detach().item()
    cls_loss = cls_loss.mean().item()
    
    return box_loss, cls_loss    
    
    
def test_step(images, true_boxes, true_classes, difficulties, model, amp,
              metrics, device):                       
    images = images.to(device)
    true_boxes = [x.to(device) for x in true_boxes]
    true_classes = [x.to(device) for x in true_classes]
    difficulties = [x.to(device) for x in difficulties]

    with autocast(enabled=amp):
        preds = model(images)
        loss, box_loss, cls_loss = model.compute_loss(preds, true_boxes, true_classes)
        #outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in preds]

    loss = loss.item()
    metrics['loss'].update(loss, images.shape[0])

    det_boxes, det_scores, det_classes = nms(*model.decode(preds))

    print(f'length of first class image : {len(det_classes[0])}')
    print(f'length of second class image : {len(det_classes[1])}')
    print(f'length of third class image : {len(det_classes[2])}')
    print(f'length of fourth class image : {len(det_classes[3])}')

    

    metrics['APs'].update(det_boxes, det_scores, det_classes,
                          true_boxes, true_classes, difficulties)
    
    box_loss = box_loss.cpu().detach().item()
    cls_loss = cls_loss.mean().item()
    
    return box_loss, cls_loss