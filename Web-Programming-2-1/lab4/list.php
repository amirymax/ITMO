<?php
$dom = new DOMDocument();
$dom->load('data.xml');
$students = $dom->getElementsByTagName('students')->item(0);
?>

<div class="container-fuild">
    <div class="card">
        <div class="card-header">
            <h1>Students' List</h1>
        </div>
        <div class="card-body">
            <table class="table">
                <thead>
                <tr class="row">
                    <th>No.</th>
                    <th>Name & Surname</th>
                    <th>Group</th>
                    <th>ISU</th>
                    <th>Change</th>
                    <th>Remove</th>
                </tr>
                </thead>
                <tbody>
                <?php
                $i = 0;
                $student = $students->getElementsByTagName('student');
                while (is_object($student->item($i++))) {
                    ?>
                    <tr class="row">
                        <td><?php echo $i ?></td>
                        <td><?php echo $student->item($i - 1)->getElementsByTagName('name')->item(0)->nodeValue ?></td>
                        <td><?php echo $student->item($i - 1)->getElementsByTagName('group')->item(0)->nodeValue ?></td>
                        <td><?php echo $student->item($i - 1)->getElementsByTagName('isu')->item(0)->nodeValue ?></td>
                        <td>
                            <a href="index.php?page_layout=update&id=<?php echo $student->item($i - 1)->getElementsByTagName('id')->item(0)->nodeValue; ?>">
                                Change</a></td>
                        <td>
                            <a onclick="return Del('<?php echo $student->item($i - 1)->getElementsByTagName('name')->item(0)->nodeValue; ?>',<?php echo $i ?> )"
                               href="index.php?page_layout=delete&id=<?php echo $student->item($i - 1)->getElementsByTagName('id')->item(0)->nodeValue; ?>">Remove</a>
                        </td>
                    </tr>
                <?php } ?>
                </tbody>
            </table>
            <a href="index.php?page_layout=create" id="add">Add</a>
        </div>
    </div>

</div>

<script>
    function Del(name, id) {
        var x = parseInt(id);
        var s = "th";
        if (x % 10 === 1) {
            s = "st";
        } else if (x % 10 === 2) {
            s = "nd";
        } else if (x % 10 === 3) {
            s = "rd";
        }
        return confirm("Are you sure to delete " + id + s + " student " +name+" ?");
    }
</script>