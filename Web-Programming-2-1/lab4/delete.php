<?php $id = $_GET['id'];
$dom = new DOMDocument();
$dom->load('data.xml');
$students = $dom->getElementsByTagName('students')->item(0);
$student = $students->getElementsByTagName('student');

$i = 0;
while (is_object($student->item($i++))) {
    $st = $student->item($i - 1)->getElementsByTagName('id')->item(0);
    $st_id = $st->nodeValue;
    if ($st_id == $id) {
        $students->removeChild($student->item($i - 1));
        break;
    }
}

$dom->formatOutput = true;
$dom->save('data.xml') or die('Error');
header('location: index.php?page_layout=list');
?>