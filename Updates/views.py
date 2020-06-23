
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
import json
import cv2
import PIL.Image
import base64
import io
import numpy as np
import sys
sys.path.insert(0, '\\home\\KLTN_TheFaceOfArtFaceParsing\\Updates\\face_parsing')
sys.path.insert(1, '\\home\\KLTN_TheFaceOfArtFaceParsing\\Updates\\foa')
from test_1 import exportImgAPI
from warp_img import geo
import timeit
# Create your views here.


def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = PIL.Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


@csrf_exempt
def index(request):
    # start = timeit.default_timer()
    # body = json.JSONDecoder().decode(request.body.decode('utf-8'))
    # # print(body['image'])
    # input_img = stringToRGB(body['image'])
    # style = body['style']
    # # # print(sys.path[0])
    # stop = timeit.default_timer()
    # print('Time preproccessing: ', stop - start)

    # start = timeit.default_timer()
    # output_img = exportImgAPI(
    #     input_img, style, cp='79999_iter.pth')
    # stop = timeit.default_timer()
    # print('Time proccessing: ', stop - start)

    # # # output_img = cv2.imread('E:\\KLTN\\cfeapi\\Updates\\face_parsing\\textures\\btexture6.jpg')
    # # # print(type(output_img))
    # start = timeit.default_timer()
    # cv2.imwrite('result.jpg', output_img[:, :, ::-1])
    with open('result.jpg', "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    stop = timeit.default_timer()
    print('Time postproccessing: ', stop - start)
    # return render(request, 'index.html', {'image': image_data})
    # print(len(image_data))

    return JsonResponse({'image': image_data})

    # return FileResponse(open('result.jpg'))


@csrf_exempt
def foa(request):
    start = timeit.default_timer()
    body = json.JSONDecoder().decode(request.body.decode('utf-8'))
    # print(body['image'])
    input_img = stringToRGB(body['image'])
    style = body['style']
    # # print(sys.path[0])
    stop = timeit.default_timer()
    print('Time preproccessing: ', stop - start)

    start = timeit.default_timer()
    output_img = geo(input_img)
    stop = timeit.default_timer()
    print('Time proccessing: ', stop - start)

    # # output_img = cv2.imread('E:\\KLTN\\cfeapi\\Updates\\face_parsing\\textures\\btexture6.jpg')
    # # print(type(output_img))
    start = timeit.default_timer()
    cv2.imwrite('result.jpg', output_img[:, :, ::-1])
    with open('result.jpg', "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    stop = timeit.default_timer()
    print('Time postproccessing: ', stop - start)
    # return render(request, 'index.html', {'image': image_data})
    # print(len(image_data))

    return JsonResponse({'image': image_data})
