<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport"
          content="width=device-width, user-scalable=no, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0">
    <!--<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous"> --->
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <link rel="stylesheet" href="style.css">
    <title>Lab 4</title>
</head>
<body>
<?php
    if(isset($_GET['page_layout'])){
        switch ($_GET['page_layout']){
            case 'list':
                require_once 'list.php';
                break;
            case 'create':
                require_once 'create.php';
                break;
            case 'update':
                require_once 'update.php';
                break;
            case 'delete':
                require_once 'delete.php';
                break;
        }
    }else{
        require_once 'list.php';
    }
?>
</body>
</html>
