<?php
$dom = new DOMDocument();
$dom->load('data.xml');
$students = $dom->getElementsByTagName('students')->item(0);
$student = $students->getElementsByTagName('student');
$index = $student->length;
$id = $student[$index - 1]->getElementsByTagName('id')->item(0)->nodeValue + 1;

/**
 * @param DOMDocument $dom
 * @param $id
 * @return DOMElement|false
 * @throws DOMException
 */
function getF(DOMDocument $dom, $id)
{
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
    return $new_student;
}

if (isset($_POST['okay'])) {
    $new_student = getF($dom, $id);

    $students->appendChild($new_student);

    $dom->formatOutput = true;
    $dom->save('data.xml') or die('Error');
    header('location: index.php?page_layout=list');
}
?>
<link rel="stylesheet" href="create.css">
<div>
    <div>
        <h2>Add a new Student</h2>
    </div>
    <div>
        <form method="POST" enctype="multipart/form-data">
            <div class="form-item">
                <label for="name">Student's Name</label>
                <input type="text" name="name" class="form-control" required placeholder="Enter the name">
            </div>
            <div class="form-item">
                <label id="group" for="group">Student's Group</label>
                <input type="number" name="group" class="form-control" required placeholder="only numbers">
            </div>
            <div class="form-item">
                <label id="isu" for="isu">Student's ISU</label>
                <input type="number" name="isu" class="form-control" required placeholder="only numbers">
            </div>
            <button name="okay" class="btn btn-success" type="submit">Add</button>
        </form>
    </div>
</div>