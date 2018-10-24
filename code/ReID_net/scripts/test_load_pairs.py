import tensorflow as tf
import numpy as np

def main():
  n_pers = 1454

  image_list = np.genfromtxt('/fastwork/luiten/mywork/data/CUHK03/number_of_images.csv', delimiter=',')
  image_list = image_list.astype(np.int32)
  num_images = tf.constant(image_list)

  rand = tf.random_uniform([7], maxval=tf.int32.max, dtype=tf.int32)
  sample_same_person = rand[0] % 2
  pers_id_1 = ((rand[1]-1) % n_pers) + 1
  cam_id_1 = ((rand[2]-1) % 2) + 1
  pers_1_n_imgs = num_images[n_pers * (cam_id_1 - 1) + pers_id_1][2]
  img_id_1 = ((rand[3] - 1) % pers_1_n_imgs) + 1

  def if_same_person():
    pers_id_2 = pers_id_1
    cam_id_2 = cam_id_1
    img_id_2 = ((rand[4] - 1) % (pers_1_n_imgs - 1)) + 1
    img_id_2 = tf.cond(img_id_2 >= img_id_1, lambda: img_id_2 + 1, lambda: img_id_2)
    return pers_id_2, cam_id_2, img_id_2

  def if_not_same_person():
    pers_id_2 = ((rand[4]-1) % (n_pers-1)) + 1
    pers_id_2 = tf.cond(pers_id_2 >= pers_id_1, lambda: pers_id_2 + 1, lambda: pers_id_2)
    cam_id_2 = ((rand[5]-1) % 2) + 1
    pers_2_n_imgs = num_images[n_pers * (cam_id_2 - 1) + pers_id_2][2]
    img_id_2 = ((rand[6] - 1) % (pers_2_n_imgs -1)) + 1
    return pers_id_2, cam_id_2, img_id_2

  pers_id_2, cam_id_2, img_id_2 = tf.cond(tf.cast(sample_same_person,tf.bool),if_same_person,if_not_same_person)

  # pair = tf.stack([sample_same_person, pers_id_1,cam_id_1,img_id_1,pers_id_2,cam_id_2,img_id_2])

  img1 = tf.as_string(pers_id_1) + "_" + tf.as_string(cam_id_1) + "_" + tf.as_string(img_id_1) + ".png"
  img2 = tf.as_string(pers_id_2) + "_" + tf.as_string(cam_id_2) + "_" + tf.as_string(img_id_2) + ".png"

  pair = tf.stack([img1, img2, tf.as_string(sample_same_person)])


  # image_reader = tf.WholeFileReader()
  # _, image_file1 = image_reader.read(img1)
  # _, image_file2 = image_reader.read(img2)

  print(image_list.shape)
  batch = tf.train.batch([pair], batch_size=3)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  tf.train.start_queue_runners(sess)
  for i in range(1):
    x = sess.run([batch])
    print(x)

if __name__ == "__main__":
  main()