## 创建实例

## 设置白名单

设为 0.0.0.0/0 就行

## 创建账户

选择 `高权限账户`，账户 `annbench`，密码 `Fantast1c`

## 创建 PASE 插件

### 连接数据库

先启动一个 pg 的 docker 镜像：

```shell
docker pull postgres
docker run --name pg_test -e POSTGRES_PASSWORD=ljq -e POSTGRES_USER=ljq  -d -p 5432:5432 postgres
docker exec -it pg_test /bin/bash
```

### 创建 PASE 插件

```shell
psql -h pgm-bp1udjb1307k73bito.pg.rds.aliyuncs.com -p 1921 -U annbench -d postgres -W
$ create extension pase;
```
