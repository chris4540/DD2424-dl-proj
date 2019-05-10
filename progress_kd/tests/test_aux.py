"""
This script is to check if basically the teacher model fits to the auxilary model
"""
import torch
from models.vgg import Vgg
from models.vgg_aux import AuxiliaryVgg

if __name__ == "__main__":
    teacher = Vgg('VGG16', batch_norm=True)
    chkpt = torch.load("vgg16bn_teacher.tar")
    teacher.load_state_dict(chkpt['state_dict'])
    aux1 = AuxiliaryVgg(teacher, 1, batch_norm=True)
    aux1.drop_teacher_subnet_blk()

    data = {
        'state_dict': aux1.state_dict()
    }
    torch.save(data, 'aux1_state.tar')

    teacher_state = torch.load("vgg16bn_teacher.tar")['state_dict']
    aux1_state = torch.load("aux1_state.tar")['state_dict']
    assert torch.equal(
        teacher_state['features.7.weight'].float().to('cpu'),
        aux1_state['features.10.weight'].float())

    # net.drop_teacher_subnet_blk()
    # print("# of params = ", get_sum_params(net))
    # for k in range(2, 6):
    #     student = AuxiliaryVgg(net, k)
    #     student.drop_teacher_subnet_blk()
    #     print("# of params = ", get_sum_params(student))
    #     net = student


