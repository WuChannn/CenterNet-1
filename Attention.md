1. 使用MSCOCO数据集训练模型，训练在中途突然停止，可在sample/coco.py文件中：
    ```
    max_tag_len = 128
    修改为：
    max_tag_len = 256
    ```
2. 直接使用由MSCOCO数据集训练得到的模型测试口算图片，由于预测出的置信度较低，因此在可视化图片中无法显示，此时可在test/coco.py文件中：
    ```
    keep_inds = (top_bboxes[image_id][j][:, -1] >= 0.4)
    修改为
    keep_inds = (top_bboxes[image_id][j][:, -1] >= 0.1)
    ```
3. 口算数据集image_file的准备
    - 直接从ks_*.txt读出图片路径：db/coco.py
    ```
    # self._image_dir  = os.path.join(self._coco_dir, "images", self._dataset)
    # self._image_file = os.path.join(self._image_dir, "{}")

    self._text_file = os.path.join(self._coco_dir, "ks_{}.txt")
    self._text_file = self._label_file.format(self._dataset)
    with open(self._text_file, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        img_files = [i.split()[0] for i in annotations]
    self._image_file = img_files
    ```
    - 修改db/base.py
    ```
    def image_file(self, ind):
        if self._image_file is None:
            raise ValueError("Image path is not initialized")

        image_id = self._image_ids[ind]
        # return self._image_file.format(image_id)
        return image_id
    ```
4. 超参数修改：
    config/CenterNet-xx.json
    ```
    "top_k": 5,
    "categories": 6,
    ```
    db/detection.py
    ```
    self._configs["categories"] = 6
    ```
    models/CenterNet-52.py
    ```
    out_dim = 6
    ```
    sample/coco.py
    ```
    max_tag_len = 256 (同1)
    ```
5. 训练
    ```
    python train.py CenterNet-52
    ```
    如下问题的出现是因为shm_size设置太小
    ~~~
    Traceback (most recent call last):
    File "/root/data/anaconda3/envs/CenterNet_duan/lib/python3.6/multiprocessing/queues.py", line 234, in _feed
        obj = _ForkingPickler.dumps(obj)
    File "/root/data/anaconda3/envs/CenterNet_duan/lib/python3.6/multiprocessing/reduction.py", line 51, in dumps
        cls(buf, protocol).dump(obj)
    File "/root/data/anaconda3/envs/CenterNet/lib/python3.6/site-packages/torch/multiprocessing/reductions.py", line 190, in reduce_storage
        fd, size = storage._share_fd_()
    RuntimeError: unable to write to file </torch_1682_2443637334>
    ~~~
    如下问题出现是因为使用了多GPU训练
    ~~~
    Exception in thread Thread-1:
    Traceback (most recent call last):
    File "/root/data/anaconda3/envs/CenterNet_duan/lib/python3.6/threading.py", line 916, in _bootstrap_inner
        self.run()
    File "/root/data/anaconda3/envs/CenterNet_duan/lib/python3.6/threading.py", line 864, in run
        self._target(*self._args, **self._kwargs)
    File "train.py", line 51, in pin_memory
        data = data_queue.get()
    File "/root/data/anaconda3/envs/CenterNet_duan/lib/python3.6/multiprocessing/queues.py", line 113, in get
        return _ForkingPickler.loads(res)
    File "/root/data/anaconda3/envs/CenterNet/lib/python3.6/site-packages/torch/multiprocessing/reductions.py", line 151, in rebuild_storage_fd
        fd = df.detach()
    File "/root/data/anaconda3/envs/CenterNet_duan/lib/python3.6/multiprocessing/resource_sharer.py", line 57, in detach
        with _resource_sharer.get_connection(self._id) as conn:
    File "/root/data/anaconda3/envs/CenterNet_duan/lib/python3.6/multiprocessing/resource_sharer.py", line 87, in get_connection
        c = Client(address, authkey=process.current_process().authkey)
    File "/root/data/anaconda3/envs/CenterNet_duan/lib/python3.6/multiprocessing/connection.py", line 487, in Client
        c = SocketClient(address)
    File "/root/data/anaconda3/envs/CenterNet_duan/lib/python3.6/multiprocessing/connection.py", line 614, in SocketClient
        s.connect(address)
    ConnectionRefusedError: [Errno 111] Connection refused
    ~~~
6. 测试
    ```
    python test.py CenterNet-52 --testiter 5000 --debug
    ```
    test/coco.py
    ```
    colours = np.random.rand(6,3)
    ```
    models/py_utils/kp_utils.py
    ```
    # num_dets need to be less than top_k * top_k
    def _decode(
        tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, ct_heat, ct_regr, 
        K=100, kernel=1, ae_threshold=1, num_dets=20
    ):
    ```
    test/coco.py
    ```
    # 该修改适合于CUDA out of memory的情况，此时config/CenterNet-52.json中的test_scales应小于1
    # 如果CUDA memory充足，应该将test_scales设置为1，并且不对此部分代码进行修改
    
    if scale == 1:
        center_points.append(center)
    修改为：
    # if scale == 1:
    center_points.append(center)
    ```
    ```
    Traceback (most recent call last):
        File "test.py", line 99, in <module>
            test(testing_db, args.split, args.testiter, args.debug, args.suffix)
        File "test.py", line 61, in test
            testing(db, nnet, result_dir, debug=debug)
        File "/root/data/ks/code/CenterNet/test/coco.py", line 328, in testing
            return globals()[system_configs.sampling_function](db, nnet, result_dir, debug=debug)
        File "/root/data/ks/code/CenterNet/test/coco.py", line 324, in kp_detection
            db.evaluate(result_json, cls_ids, image_ids)
        File "/root/data/ks/code/CenterNet/db/coco.py", line 184, in evaluate
            coco_dets = coco.loadRes(result_json)
        File "data/coco/PythonAPI/pycocotools/coco.py", line 318, in loadRes
            if 'caption' in anns[0]:
    IndexError: list index out of range

    出现上述问题的原因是在测试的所有图片中未检测到bounding box，也即模型预测错误
    ```