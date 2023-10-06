<?php
$id = $_GET['id'];
$dom = new DOMDocument();
$dom->load('data.xml');
$students = $dom->getElementsByTagName('students')->item(0);
$student = $students->getElementsByTagName('student');

if (isset($_POST['okay'])) {
    $name = $_POST['name'];
    $group = $_POST['group'];
    $isu = $_POST['isu'];
    $new_student = $dom->createElement('student');

    $st_id = $dom->createElement('id', $id);
    $new_student->appendChild($st_id);

    $st_name = $dom->createElement('name', $name);
    $new_student->appendChild($st_name);

    $st_group = $dom->createElement('group', $group);
    $new_student->appendChild($st_group);

    $st_isu = $dom->createElement('isu', $isu);
    $new_student->appendChild($st_isu);
    $i = 0;
    while (is_object($student->item($i++))) {
        $std = $student->item($i - 1)->getElementsByTagName('id')->item(0);
        $std_id = $std->nodeValue;
        if ($std_id == $id) {
            $students->replaceChild($new_student, $student->item($i - 1));
            break;
        }
    }

    $dom->formatOutput = true;
    $dom->save('data.xml') or die('Error');
    header('location: index.php?page_layout=list');
}
?>

<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport"
          content="width=device-width, user-scalable=no, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Document</title>
    <link rel="stylesheet" href="update.css">
</head>
<body>
<div class="container-fluid">

    <div>
        <h2>Edit Student's Data</h2>
    </div>
    <div>
        <form method="POST" enctype="multipart/form-data">
            <div class="form-item">
                <label for="name">Student's Name</label>
                <input type="text" name="name" required placeholder="Enter the name">
            </div>
            <div class="form-item">
                <label id="group" for="group">Student's Group</label>
                <input type="number" name="group" class="form-control" required placeholder="only numbers">
            </div>
            <div class="form-item">
                <label id="isu" for="isu">Student's ISU</label>
                <input type="number" name="isu" class="form-control" required placeholder="only numbers">
            </div>
            <button name="okay" class="btn btn-success" type="submit">Update</button>
        </form>
    </div>

</div>
</body>
</html>