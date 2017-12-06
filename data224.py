import numpy as np
from PIL import Image


for i in range(1,94):
    input_filename = 'All247images/JPCNN'+str(i).zfill(3)+'.IMG'
    shape = (2048, 2048) # matrix size
    dtype = np.dtype('>u2') # big-endian unsigned integer (16bit)
#    output_filename = 'All247jpg_zxq/JPCNN'+str(i).zfill(3)+'.JPG'

# Reading.
    fid = open(input_filename, 'rb')
    data = np.fromfile(fid, dtype)
    data = np.floor(data/16)
    image = data.reshape(shape)
    image1 = Image.fromarray(image).convert('L')

    for k in range(1,6):
        n = np.floor(np.random.rand()*69+256)
        n = n.astype(int)
        image2 = image1.resize([n,n])
        left = (n-224)/2
        box = (left,left,left+224,left+224)
        image3 = image2.crop(box)
        imrgb3 = image3.convert('RGB')
        im1 = imrgb3.split()
        im1[0].show
        im1[1].show
        im1[2].show
        imrgb3.save('All247jpg_224_3D/N/JPCNN'+str(i).zfill(3)+'_'+str(2*(k-1))+'.JPG')
#        image3.show()
#        image3.save('All247jpg_224/JPCNN'+str(i).zfill(3)+'_'+str(2*(k-1))+'.JPG')

        image4 = image3.transpose(Image.FLIP_LEFT_RIGHT)
        imrgb4 = image4.convert('RGB')
        im1 = imrgb4.split()
        im1[0].show
        im1[1].show
        im1[2].show
        imrgb4.save('All247jpg_224_3D/N/JPCNN'+str(i).zfill(3)+'_'+str(2*k-1)+'.JPG')

        print(k)
#		plt.imshow(image) #Needs to be in row,col order
#		plt.savefig(output_filename)
    print(i)

for j in range(1,155):
    
    input_filename = 'All247images/JPCLN'+str(j).zfill(3)+'.IMG'
    shape = (2048, 2048) # matrix size
    dtype = np.dtype('>u2') # big-endian unsigned integer (16bit)
 #   output_filename = 'All247jpg_zxq/JPCLN'+str(j).zfill(3)+'.JPG'
    fid = open(input_filename, 'rb')
    data = np.fromfile(fid, dtype)
    data = np.floor(data/16)
    image = data.reshape(shape)
    image1 = Image.fromarray(image).convert('L')
#		np.savetxt('All247images\ALL247jpg\JPCLN'+str(i).zfill(3)+'.txt',image)
    for k in range(1,6):
        n = np.floor(np.random.rand()*69+256)
        n = n.astype(int)
        image2 = image1.resize([n,n])
        left = (n-224)/2
        box = (left,left,left+224,left+224)
        image3 = image2.crop(box)
        imrgb3 = image3.convert('RGB')
        im1 = imrgb3.split()
        im1[0].show
        im1[1].show
        im1[2].show
        imrgb3.save('All247jpg_224_3D/L/JPCLN'+str(j).zfill(3)+'_'+str(2*(k-1))+'.JPG')


        image4= image3.transpose(Image.FLIP_LEFT_RIGHT)
        imrgb34 = image4.convert('RGB')
        im1 = imrgb4.split()
        im1[0].show
        im1[1].show
        im1[2].show
        imrgb4.save('All247jpg_224_3D/L/JPCLN'+str(j).zfill(3)+'_'+str(2*k-1)+'.JPG')


        print(k)    
#		plt.imshow(image) #Needs to be in row,col order
	#	plt.savefig(output_filename)
    print(j)




